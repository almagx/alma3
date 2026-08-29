from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .clinical_result import results_from_logits
from .download import load_release
from .sitewise import real_coverage_presentation


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index is unavailable: {index}")
        return torch.device("cuda", index)
    return device


def _require_batch_size(value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("batch_size must be a positive integer")
    return value


class ALMA3:
    """Load one ALMA3-Dx release and reuse it for one or many samples."""

    def __init__(self, artifact: str | Path | None = None, *, device: str = "auto") -> None:
        """Load a release from an explicit path, configured path, cache, or verified download."""

        self.device = resolve_device(device)
        validated = load_release(artifact, device=str(self.device))
        self.artifact = validated["root"]
        self._validated = validated
        self.model = validated["model"]
        self.cpg = validated["cpg"]
        self.taxonomy = validated["taxonomy"]
        self.thresholds = validated["thresholds"]
        hashes = validated["hashes"]
        self.release_identity = {
            "model_sha256": hashes["model.safetensors"],
            "taxonomy_sha256": hashes["taxonomy.json"],
            "thresholds_sha256": hashes["thresholds.json"],
        }
        self.sidecar_release_identity = {
            "manifest_sha256": validated["manifest_sha256"],
            "model_sha256": hashes["model.safetensors"],
            "taxonomy_sha256": hashes["taxonomy.json"],
            "cpg_manifest_sha256": hashes["cpg_manifest.json"],
            "thresholds_sha256": hashes["thresholds.json"],
        }
        self.minimum_observed_cpgs = int(self.thresholds["minimum_observed_cpgs"])
        self._chr_id = self.cpg.chr_id[None, :].to(self.device)
        self._pos = self.cpg.pos[None, :].to(self.device)

    @property
    def validated_release(self) -> dict[str, Any]:
        return self._validated

    def _predict_tensors(
        self,
        sample_ids: Sequence[str],
        beta: torch.Tensor,
        observed: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> tuple[list[dict[str, Any]], torch.Tensor, list[int]]:
        ids = [str(value) for value in sample_ids]
        if not ids or any(not value.strip() for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("sample IDs must be nonempty and unique")
        if beta.ndim != 2 or beta.shape != observed.shape or beta.shape != uncertainty.shape:
            raise ValueError("beta, observed, and uncertainty must have the same two-dimensional shape")
        if beta.shape[0] != len(ids) or beta.shape[1] != len(self.cpg.cpg_ids):
            raise ValueError("sample tensors do not match sample IDs or the release CpG manifest")
        observed_counts = [int(value) for value in observed.sum(dim=1).tolist()]
        too_sparse = [
            f"{sample_id}:{count}"
            for sample_id, count in zip(ids, observed_counts, strict=True)
            if count < self.minimum_observed_cpgs
        ]
        if too_sparse:
            from .infer import InputContractError

            raise InputContractError(
                "inference sample below calibrated observed-CpG floor "
                f"{self.minimum_observed_cpgs}: {too_sparse[:5]}"
            )
        with torch.inference_mode():
            observed_device = observed.to(self.device)
            embedding = self.model.embed(
                beta.to(self.device),
                observed_device,
                uncertainty.to(self.device),
                self._chr_id.expand(beta.shape[0], -1),
                self._pos.expand(beta.shape[0], -1),
            )
            logits = self.model.logits_from_embedding(embedding)
            results = results_from_logits(
                ids,
                logits,
                self.thresholds,
                self.taxonomy,
                self.release_identity,
            )
        return results, embedding, observed_counts

    def predict_bedmethyl(
        self,
        inputs: str | Path | Sequence[str | Path],
        *,
        batch_size: int = 1,
    ) -> list[dict[str, Any]]:
        """Predict one sample per BedMethyl file while preserving the supplied file order."""

        from .infer import load_bed_methyl_with_manifest

        size = _require_batch_size(batch_size)
        paths = [Path(inputs)] if isinstance(inputs, (str, Path)) else [Path(value) for value in inputs]
        if not paths:
            raise ValueError("predict_bedmethyl requires at least one input")
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        batch_ids: list[str] = []
        batch_beta: list[torch.Tensor] = []
        batch_observed: list[torch.Tensor] = []
        batch_uncertainty: list[torch.Tensor] = []

        def flush() -> None:
            if not batch_ids:
                return
            batch_results, _, _ = self._predict_tensors(
                batch_ids,
                torch.cat(batch_beta),
                torch.cat(batch_observed),
                torch.cat(batch_uncertainty),
            )
            results.extend(batch_results)
            batch_ids.clear()
            batch_beta.clear()
            batch_observed.clear()
            batch_uncertainty.clear()

        for path in paths:
            sample_ids, beta, observed, coverage = load_bed_methyl_with_manifest(path, self.cpg)
            sample_id = sample_ids[0]
            if sample_id in seen:
                raise ValueError(f"duplicate BedMethyl sample ID: {sample_id}")
            seen.add(sample_id)
            presentation = real_coverage_presentation(beta, observed, coverage)
            batch_ids.append(sample_id)
            batch_beta.append(presentation.beta_input)
            batch_observed.append(presentation.input_observed)
            batch_uncertainty.append(presentation.uncertainty)
            if len(batch_ids) == size:
                flush()
        flush()
        return results

    def predict_array(
        self,
        beta: Any,
        cpg_ids: Sequence[str],
        sample_ids: Sequence[str] | None = None,
        *,
        batch_size: int = 1,
    ) -> list[dict[str, Any]]:
        """Predict rows of beta values aligned by CpG ID; NaN values are unobserved."""

        size = _require_batch_size(batch_size)
        columns = [str(value) for value in cpg_ids]
        if not columns or any(not value for value in columns) or len(columns) != len(set(columns)):
            raise ValueError("cpg_ids must be nonempty and unique")
        try:
            values = torch.as_tensor(beta, dtype=torch.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("beta must be a numeric one- or two-dimensional array") from error
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != len(columns) or values.shape[0] == 0:
            raise ValueError("beta shape must be [samples, len(cpg_ids)]")
        if bool(torch.isinf(values).any().item()):
            raise ValueError("beta values must not contain infinity")
        finite = torch.isfinite(values)
        if bool(((values[finite] < 0) | (values[finite] > 1)).any().item()):
            raise ValueError("observed beta values must be in [0, 1]")
        ids = (
            [f"sample-{index + 1}" for index in range(values.shape[0])]
            if sample_ids is None
            else [str(value) for value in sample_ids]
        )
        if len(ids) != values.shape[0] or any(not value.strip() for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("sample_ids must be nonempty, unique, and match beta rows")

        release_index = {name: index for index, name in enumerate(self.cpg.cpg_ids)}
        aligned = torch.zeros((values.shape[0], len(self.cpg.cpg_ids)), dtype=torch.float32)
        observed = torch.zeros_like(aligned, dtype=torch.bool)
        for source_index, name in enumerate(columns):
            target_index = release_index.get(name)
            if target_index is None:
                continue
            column_observed = finite[:, source_index]
            aligned[:, target_index] = torch.where(column_observed, values[:, source_index], 0.0)
            observed[:, target_index] = column_observed

        results: list[dict[str, Any]] = []
        uncertainty = torch.zeros_like(aligned)
        for start in range(0, len(ids), size):
            stop = min(start + size, len(ids))
            batch_results, _, _ = self._predict_tensors(
                ids[start:stop],
                aligned[start:stop],
                observed[start:stop],
                uncertainty[start:stop],
            )
            results.extend(batch_results)
        return results

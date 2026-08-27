from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_ALMA3_CHROMOSOME_COUNT = 24
_GRCH38_CENTROMERE_BOUNDARIES = (
    123_400_000,
    93_900_000,
    90_900_000,
    50_000_000,
    48_800_000,
    59_800_000,
    60_100_000,
    45_200_000,
    43_000_000,
    39_800_000,
    53_400_000,
    35_500_000,
    17_700_000,
    17_200_000,
    19_000_000,
    36_800_000,
    25_100_000,
    18_500_000,
    26_200_000,
    28_100_000,
    12_000_000,
    15_000_000,
    61_000_000,
)


class DataContractError(ValueError):
    """Raised when an inference input violates the ALMA3 data contract."""


@dataclass(frozen=True)
class CpGManifest:
    cpg_ids: tuple[str, ...]
    chr_id: torch.Tensor
    pos: torch.Tensor
    source_cpg_manifest_sha256: str | None = None
    arm_id: torch.Tensor | None = None

    @classmethod
    def load(cls, path: str | Path) -> "CpGManifest":
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        cpg_ids = tuple(str(value) for value in payload.get("cpg_ids", ()))
        if not cpg_ids:
            raise DataContractError("CpG manifest requires non-empty cpg_ids")
        if len(set(cpg_ids)) != len(cpg_ids):
            raise DataContractError("CpG manifest cpg_ids must be unique")
        declared_digest = payload.get("cpg_manifest_sha256")
        if declared_digest is not None:
            calculated = hashlib.sha256(("\n".join(cpg_ids) + "\n").encode()).hexdigest()
            if str(declared_digest).lower() != calculated:
                raise DataContractError("CpG manifest SHA256 does not match its ordered cpg_ids")
        chr_raw = payload.get("chr_id")
        pos_raw = payload.get("pos")
        chrom_raw = payload.get("chrom") or payload.get("chr")
        start_raw = payload.get("start")
        end_raw = payload.get("end")
        if chr_raw is None:
            if chrom_raw is None:
                raise DataContractError("CpG manifest requires chr_id or chrom arrays")
            chr_raw = [_chrom_to_id(value) for value in chrom_raw]
        if pos_raw is None:
            raise DataContractError("architecture-5 CpG manifest requires explicit pos values")
        if not isinstance(chr_raw, list) or any(type(value) is not int for value in chr_raw):
            raise DataContractError("CpG manifest chr_id values must be integers")
        if not isinstance(pos_raw, list) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in pos_raw
        ):
            raise DataContractError("CpG manifest pos values must be numeric")
        if len(chr_raw) != len(cpg_ids) or len(pos_raw) != len(cpg_ids):
            raise DataContractError("CpG manifest cpg_ids, chr_id, and pos lengths must match")
        chr_id = torch.as_tensor(chr_raw, dtype=torch.long)
        pos = torch.as_tensor(pos_raw, dtype=torch.float32)
        if not bool(torch.isfinite(pos).all().item()) or bool(((pos < 0) | (pos > 1)).any().item()):
            raise DataContractError("CpG manifest pos values must be finite and normalized to [0, 1]")
        if bool(((chr_id < 0) | (chr_id >= _ALMA3_CHROMOSOME_COUNT)).any().item()):
            raise DataContractError("CpG manifest chr_id values must be in [0, 23]")
        if chr_id.numel() > 1:
            descending = (chr_id[1:] < chr_id[:-1]) | (
                (chr_id[1:] == chr_id[:-1]) & (pos[1:] < pos[:-1])
            )
            if bool(descending.any().item()):
                raise DataContractError("CpG order must follow nondecreasing genomic coordinate order")
        arm_id = (
            _arm_ids_from_intervals(chrom_raw, start_raw, end_raw, chr_id)
            if chrom_raw is not None and start_raw is not None and end_raw is not None
            else None
        )
        source_hash = payload.get("source_cpg_manifest_sha256")
        return cls(
            cpg_ids=cpg_ids,
            chr_id=chr_id,
            pos=pos,
            source_cpg_manifest_sha256=str(source_hash).lower() if source_hash else None,
            arm_id=arm_id,
        )


def _chrom_to_id(value: object) -> int:
    text = str(value).lower().removeprefix("chr")
    if text == "x":
        return 22
    if text == "y":
        return 23
    try:
        chromosome = int(text)
    except ValueError:
        raise DataContractError(f"invalid chromosome value: {value}") from None
    if not 1 <= chromosome <= 22:
        raise DataContractError(f"invalid chromosome value: {value}")
    return chromosome - 1


def _arm_ids_from_intervals(
    chrom: Any,
    start: Any,
    end: Any,
    expected_chr_id: torch.Tensor,
) -> torch.Tensor:
    if not all(isinstance(values, list) for values in (chrom, start, end)):
        raise DataContractError("CpG manifest chrom, start, and end must be arrays")
    if not (len(chrom) == len(start) == len(end) == int(expected_chr_id.numel())):
        raise DataContractError("CpG manifest chrom, start, and end lengths must match cpg_ids")
    arm_ids = []
    for index, (raw_chrom, raw_start, raw_end) in enumerate(zip(chrom, start, end, strict=True)):
        chromosome = _chrom_to_id(raw_chrom)
        if chromosome != int(expected_chr_id[index]):
            raise DataContractError("CpG manifest chrom and chr_id arrays disagree")
        if type(raw_start) is not int or type(raw_end) is not int or raw_start < 0 or raw_end <= raw_start:
            raise DataContractError("CpG manifest intervals require non-negative start and end > start")
        if chromosome == 23:
            raise DataContractError("CpG arm layout must exclude chrY")
        boundary = _GRCH38_CENTROMERE_BOUNDARIES[chromosome]
        if raw_end <= boundary:
            arm_ids.append(2 * chromosome)
        elif raw_start >= boundary:
            arm_ids.append(2 * chromosome + 1)
        else:
            raise DataContractError(
                "CpG interval straddles the GRCh38 centromere boundary: "
                f"{raw_chrom}:{raw_start}-{raw_end}"
            )
    return torch.as_tensor(arm_ids, dtype=torch.long)

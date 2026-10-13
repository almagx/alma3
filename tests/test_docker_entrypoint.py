from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker-entrypoint.sh"


class DockerEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env = {**os.environ, "PATH": f"{self.root}:{os.environ['PATH']}"}
        self.executable(
            "alma3",
            f"#!{sys.executable}\nimport json, sys\n"
            "print(json.dumps([sys.argv[1:], sys.stdin.read()]))\n",
        )

    def executable(self, name: str, source: str) -> None:
        path = self.root / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)

    def run_entrypoint(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["sh", str(ENTRYPOINT), *arguments], cwd=self.root, env=self.env,
            input="standard input\n", text=True, capture_output=True, timeout=10, check=False,
        )

    def test_dispatch_preserves_arguments_and_standard_input(self) -> None:
        for arguments in (
            (), ("demo",), ("infer", "-i", "private sample.csv.gz"),
            ("verify-release",), ("export-bedmethyl-target",),
            ("-h",), ("--help",), ("--version",), ("alma3", "--version"),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_entrypoint(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = list(arguments[1:] if arguments[:1] == ("alma3",) else arguments)
                self.assertEqual(json.loads(result.stdout), [expected or ["--help"], "standard input\n"])

    def test_custom_command_preserves_failure_and_stderr(self) -> None:
        result = self.run_entrypoint("sh", "-c", "echo command-failed >&2; exit 37")
        self.assertEqual(result.returncode, 37)
        self.assertEqual(result.stderr, "command-failed\n")

    def test_non_root_user_is_preserved(self) -> None:
        self.executable("id", "#!/bin/sh\necho 1234\n")
        self.executable("stat", "#!/bin/sh\nexit 71\n")
        self.executable("setpriv", "#!/bin/sh\nexit 72\n")
        result = self.run_entrypoint("demo")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_privilege_setup_stops_execution(self) -> None:
        self.executable("id", "#!/bin/sh\necho 0\n")
        self.executable("setpriv", "#!/bin/sh\necho privilege-setup-failed >&2; exit 73\n")
        for owner in ("0:0", "1234:5678"):
            with self.subTest(owner=owner):
                self.executable("stat", f"#!/bin/sh\necho {owner}\n")
                result = self.run_entrypoint("demo")
                self.assertEqual(result.returncode, 73)
                self.assertEqual(result.stderr, "privilege-setup-failed\n")
                self.assertEqual(result.stdout, "")

    def test_interruptions_reach_the_command(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=sig):
                code = (
                    "import signal, sys; "
                    f"signal.signal({int(sig)}, lambda *_: sys.exit(42)); "
                    "print('ready', flush=True); signal.pause()"
                )
                with subprocess.Popen(
                    ["sh", str(ENTRYPOINT), sys.executable, "-c", code],
                    cwd=self.root, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                ) as process:
                    try:
                        with selectors.DefaultSelector() as selector:
                            selector.register(process.stdout, selectors.EVENT_READ)
                            self.assertTrue(selector.select(timeout=10), "command did not start")
                        self.assertEqual(process.stdout.readline(), b"ready\n")
                        process.send_signal(sig)
                        process.communicate(timeout=10)
                        self.assertEqual(process.returncode, 42)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.communicate()


if __name__ == "__main__":
    unittest.main()

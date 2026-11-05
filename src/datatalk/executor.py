"""Subprocess execution with per-attempt files and wall-clock timeout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from .models import ExecutionResult
from .policy import validate_code


class PythonExecutor:
    def __init__(self, timeout_seconds: float = 30) -> None:
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str, df: pd.DataFrame, output_dir: Path) -> ExecutionResult:
        started = time.monotonic()
        try:
            validate_code(code)
        except (ValueError, SyntaxError) as exc:
            return ExecutionResult(success=False, error=f"Code validation failed: {exc}")
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analysis.py").write_text(code, encoding="utf-8")
        # Preserve platform variables required by Python while excluding API credentials.
        allowed = {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL"}
        child_env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        child_env.update(
            {
                "MPLCONFIGDIR": str(output_dir / ".matplotlib"),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
        )
        with tempfile.TemporaryDirectory(prefix="datatalk-") as directory:
            data_path = Path(directory) / "data.pkl"
            # Only the parent serializes input; user-provided pickle files are never loaded.
            df.to_pickle(data_path)
            payload = json.dumps({"code": code, "data_path": str(data_path)}, ensure_ascii=True)
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-m", "datatalk._worker"],
                    input=payload,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=output_dir,
                    env=child_env,
                    timeout=self.timeout_seconds,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    success=False,
                    error=f"Execution timed out after {self.timeout_seconds:g} seconds",
                    duration_seconds=time.monotonic() - started,
                )
        duration = time.monotonic() - started
        if completed.returncode:
            return ExecutionResult(
                success=False,
                error=f"Worker exited with code {completed.returncode}: {completed.stderr[-2000:]}",
                duration_seconds=duration,
            )
        try:
            result = json.loads(completed.stdout)
            artifacts = []
            for filename in result.get("artifacts", []):
                path = Path(filename).resolve()
                if path.parent != output_dir or path.suffix != ".png" or not path.is_file():
                    raise ValueError("Worker returned an invalid chart path")
                artifacts.append(str(path))
            return ExecutionResult(
                success=bool(result["success"]),
                stdout=result.get("stdout", ""),
                error=result.get("error"),
                artifacts=artifacts,
                duration_seconds=duration,
            )
        except (ValueError, KeyError, TypeError):
            return ExecutionResult(
                success=False,
                error="Worker returned an invalid result",
                duration_seconds=duration,
            )

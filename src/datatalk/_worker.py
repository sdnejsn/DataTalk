"""Single-use analysis process. Configuration and API keys are not passed here."""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import re
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from datatalk.policy import ALLOWED_MODULES, is_prohibited_attribute, validate_code  # noqa: E402

MAX_OUTPUT = 32_000


class BoundedOutput(io.StringIO):
    def write(self, value: str) -> int:
        remaining = MAX_OUTPUT - self.tell()
        if remaining > 0:
            super().write(value[:remaining])
        return len(value)


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in ALLOWED_MODULES:
        raise ImportError(f"Module is not permitted: {name}")
    if any(is_prohibited_attribute(part) or part == "*" for part in fromlist):
        raise ImportError("Private, file-access and wildcard imports are prohibited")
    return builtins.__import__(name, globals, locals, fromlist, level)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    payload = json.load(sys.stdin)
    output = BoundedOutput()
    error = None
    artifacts: list[str] = []
    try:
        validate_code(payload["code"])
        df = pd.read_pickle(payload["data_path"])
        font_names = {font.name for font in font_manager.fontManager.ttflist}
        candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
        plt.rcParams["font.sans-serif"] = [name for name in candidates if name in font_names]
        plt.rcParams["axes.unicode_minus"] = False
        original_savefig = matplotlib.figure.Figure.savefig

        def checked_savefig(figure, fname, *args, **kwargs):
            filename = str(fname)
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\.png", filename):
                raise ValueError("Use a plain PNG filename such as output.png")
            if len(artifacts) >= 5:
                raise ValueError("At most five chart files are permitted per attempt")
            original_savefig(figure, filename, *args, **kwargs)
            if filename not in artifacts:
                artifacts.append(filename)

        matplotlib.figure.Figure.savefig = checked_savefig
        names = (
            "abs all any bool dict enumerate filter float format frozenset int isinstance "
            "issubclass iter len list map max min next print range repr reversed round "
            "set slice sorted str sum tuple zip Exception ValueError TypeError KeyError "
            "RuntimeError ZeroDivisionError IndexError NotImplementedError"
        ).split()
        scope = {
            "__builtins__": {
                **{name: getattr(builtins, name) for name in names},
                "__import__": safe_import,
            },
            "df": df,
            "pd": pd,
            "np": np,
            "plt": plt,
        }
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exec(compile(payload["code"], "<analysis>", "exec"), scope, scope)
        if not output.getvalue().strip() and not artifacts:
            raise ValueError("Code produced no printed answer or saved chart")
    except Exception:
        error = traceback.format_exc(limit=5)[-8000:]
    finally:
        plt.close("all")
    result = {
        "success": error is None,
        "stdout": output.getvalue(),
        "error": error,
        "artifacts": [str(Path(name).resolve()) for name in artifacts] if error is None else [],
    }
    sys.stdout.write(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()

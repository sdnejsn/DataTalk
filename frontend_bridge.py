import atexit
import threading
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class FrontendBridge:
    def __init__(self, engine_factory=None):
        self._engine_factory = engine_factory
        self._engine = None
        self._engine_lock = threading.Lock()
        atexit.register(self.close)

    def close(self):
        with self._engine_lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None

    def _get_engine(self):
        with self._engine_lock:
            if self._engine is None:
                if self._engine_factory is None:
                    from datatalk import AnalysisEngine, Settings

                    self._engine = AnalysisEngine(Settings.from_env(".env"))
                else:
                    self._engine = self._engine_factory()
            return self._engine

    @staticmethod
    def read_csv(file_path):
        if not file_path:
            raise ValueError("Select a CSV file to continue.")
        path = Path(file_path)
        if path.suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported.")
        if not path.is_file():
            raise ValueError("The uploaded CSV file could not be found.")
        if path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("The CSV file must not exceed 20 MiB.")
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                frame = pd.read_csv(path, encoding="gb18030")
            except (UnicodeError, OSError, ValueError, pd.errors.ParserError) as exc:
                raise ValueError(f"Cannot read this CSV file: {exc}") from exc
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise ValueError(f"Cannot read this CSV file: {exc}") from exc
        if frame.empty or len(frame.columns) == 0:
            raise ValueError("The dataset must have at least one row and one column.")
        if len(frame) > 100000 or len(frame.columns) > 200:
            raise ValueError("The dataset exceeds the backend row or column limit.")
        if any(not isinstance(c, str) or not c.strip() for c in frame.columns):
            raise ValueError("Every column must have a nonempty text name.")
        if not frame.columns.is_unique:
            raise ValueError("Column names must be unique.")
        return frame

    @staticmethod
    def schema(frame):
        if frame is None:
            return pd.DataFrame(columns=["Column", "Type", "Missing"])
        return pd.DataFrame(
            {
                "Column": [str(column) for column in frame.columns],
                "Type": [str(dtype) for dtype in frame.dtypes],
                "Missing": [int(n) for n in frame.isna().sum()],
            }
        )

    @staticmethod
    def _validate_question(question, frame):
        if frame is None:
            raise ValueError("Upload a CSV dataset before running an analysis.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a question before running an analysis.")
        if len(question.strip()) > 8000:
            raise ValueError("Questions must be at most 8,000 characters long.")
        return question.strip()

    @staticmethod
    def _format_result(result):
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        if not isinstance(payload, dict):
            raise TypeError("The backend returned an unsupported result object.")
        success = bool(payload.get("success", False))
        error = payload.get("error")
        stdout = str(payload.get("stdout") or "")
        run_id = str(payload.get("run_id") or "")
        provider = str(payload.get("provider") or "Unknown")
        model = str(payload.get("model") or "Unknown")
        artifacts = payload.get("artifacts") or []
        if not isinstance(artifacts, (list, tuple)):
            artifacts = []
        plots = [str(path) for path in artifacts if Path(str(path)).is_file() and Path(str(path)).suffix.lower() == ".png"]
        status = "Completed" if success else "Failed"
        metadata = f"{status} | {provider} / {model} | Run: {run_id[:8] or 'N/A'}"
        if success:
            text = stdout.strip() or ("The analysis generated a chart." if plots else "Execution completed.")
        else:
            text = f"Analysis failed: {error or 'No error details provided.'}"
        return {
            "success": success,
            "message": text,
            "code": str(payload.get("code") or ""),
            "plots": plots,
            "status": metadata,
            "attempts": payload.get("attempts") or [],
            "pass_at_1": payload.get("pass_at_1"),
            "retry_count": payload.get("retry_count"),
        }

    def analyze(self, question, frame):
        question = self._validate_question(question, frame)
        result = self._get_engine().analyze(question=question, df=frame.copy(deep=True))
        return self._format_result(result)

    @staticmethod
    def offline_demo(question, frame):
        FrontendBridge._validate_question(question, frame)
        numeric = frame.select_dtypes(include="number")
        if numeric.empty:
            summary = f"Rows: {len(frame):,}\nColumns: {len(frame.columns):,}\nNo numeric columns are available for a summary chart."
            return {
                "success": True,
                "message": summary,
                "code": 'print(f"Rows: {len(df)}")\nprint(f"Columns: {len(df.columns)}")',
                "plots": [],
                "status": "Offline demo | Local descriptive summary | No model request",
                "attempts": [],
                "pass_at_1": None,
                "retry_count": 0,
            }
        totals = numeric.sum(numeric_only=True)
        selected = totals.head(8)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.bar([str(label) for label in selected.index], selected.to_numpy(dtype=float))
        ax.set_ylabel("Total")
        ax.set_title("Numeric column totals")
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        import tempfile

        output_dir = Path(tempfile.mkdtemp(prefix="datatalk-demo-"))
        chart_path = output_dir / "summary.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        summary = "Offline demo only. The question was not interpreted by an AI model.\n\n"
        summary += f"Rows: {len(frame):,} | Columns: {len(frame.columns):,}\n\nNumeric totals:\n{totals.to_string()}"
        return {
            "success": True,
            "message": summary,
            "code": 'numeric = df.select_dtypes(include="number")\nprint(numeric.sum(numeric_only=True))',
            "plots": [str(chart_path)],
            "status": "Offline demo | Local descriptive summary | No model request",
            "attempts": [],
            "pass_at_1": None,
            "retry_count": 0,
        }

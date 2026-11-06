from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app import build_app, handle_analysis, handle_upload
from frontend_bridge import FrontendBridge


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.closed = False

    def analyze(self, question, df):
        self.calls.append((question, df.copy()))
        return SimpleNamespace(
            success=True,
            stdout="sales total: 3900",
            artifacts=[],
            code='print(df["sales"].sum())',
            error=None,
            attempts=[],
            pass_at_1=True,
            retry_count=0,
            run_id="abcdef123456",
            provider="mock",
            model="fake-model",
            to_dict=lambda: {
                "success": True,
                "stdout": "sales total: 3900",
                "artifacts": [],
                "code": 'print(df["sales"].sum())',
                "error": None,
                "attempts": [],
                "pass_at_1": True,
                "retry_count": 0,
                "run_id": "abcdef123456",
                "provider": "mock",
                "model": "fake-model",
            },
        )

    def close(self):
        self.closed = True


def test_load_csv_and_schema(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("category,sales,profit\nFood,700,200\nElectronics,2000,400\n", encoding="utf-8")
    frame = FrontendBridge.read_csv(path)
    assert list(frame.columns) == ["category", "sales", "profit"]
    assert len(frame) == 2
    assert list(FrontendBridge.schema(frame)["Type"]) == ["object", "int64", "int64"]


def test_validation_and_real_engine_contract():
    engine = FakeEngine()
    adapter = FrontendBridge(engine_factory=lambda: engine)
    data = pd.DataFrame({"sales": [900, 3000]})
    output = adapter.analyze("  Sum sales  ", data)
    assert engine.calls[0][0] == "Sum sales"
    assert engine.calls[0][1].equals(data)
    assert output["success"] is True
    assert output["status"] == "Completed | mock / fake-model | Run: abcdef12"
    assert output["code"] == 'print(df["sales"].sum())'
    adapter.close()
    assert engine.closed is True


def test_backend_failure_and_artifact_filter(tmp_path):
    png = tmp_path / "output.png"
    png.write_bytes(b"fake")
    output = FrontendBridge._format_result(
        {"success": True, "stdout": "", "artifacts": [str(png), str(tmp_path / "missing.png")], "code": "", "provider": "mock"}
    )
    assert output["plots"] == [str(png)]
    error = FrontendBridge._format_result({"success": False, "error": "No tool call", "artifacts": []})
    assert "No tool call" in error["message"]


def test_offline_demo_is_labeled(tmp_path):
    data = pd.DataFrame({"sales": [5, 8], "profit": [2, 3]})
    result = FrontendBridge.offline_demo("sum sales", data)
    assert result["success"]
    assert "question was not interpreted" in result["message"]
    assert Path(result["plots"][0]).is_file()


def test_callbacks_and_gradio_build(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("sales,profit\n10,3\n20,8\n", encoding="utf-8")
    frame, preview, metadata, status, button = handle_upload(str(path))
    assert len(preview) == 2
    assert len(metadata) == 2
    assert "Dataset ready" in status
    outputs = handle_analysis("Summarize", frame, "Offline demo", [])
    assert len(outputs) == 7
    assert outputs[0][0]["role"] == "user"
    assert outputs[0][1]["role"] == "assistant"
    demo = build_app()
    assert demo is not None
    demo.close()


def test_missing_file_and_missing_question():
    with pytest.raises(ValueError, match="Select a CSV"):
        FrontendBridge.read_csv(None)
    with pytest.raises(ValueError, match="Enter a question"):
        FrontendBridge._validate_question("", pd.DataFrame({"a": [1]}))

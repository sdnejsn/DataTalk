from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pandas as pd
import pytest
from openai import OpenAI

from datatalk import AnalysisEngine, Settings, provider_catalog
from datatalk.dataset import describe_dataframe, load_csv
from datatalk.executor import PythonExecutor
from datatalk.llm import Generation, ModelError, OpenAICompatibleGenerator
from datatalk.policy import validate_code


class ScriptedGenerator:
    def __init__(self, *codes: str):
        self.codes = iter(codes)
        self.messages = []

    def generate(self, messages):
        self.messages.append(list(messages))
        code = next(self.codes)
        call_id = f"call-{len(self.messages)}"
        return Generation(
            code,
            call_id,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "execute_python",
                            "arguments": json.dumps({"code": code}),
                        },
                    }
                ],
            },
        )


@pytest.fixture
def df():
    return pd.DataFrame({"category": ["A", "A", "B"], "sales": [10, 20, 40]})


def test_real_execution_and_chart(tmp_path, df):
    code = (
        "totals = df.groupby('category')['sales'].sum()\n"
        "print('Total:', df['sales'].sum())\n"
        "totals.plot.bar()\nplt.tight_layout()\nplt.savefig('output.png')\n"
    )
    generator = ScriptedGenerator(code)
    with AnalysisEngine(Settings(output_dir=tmp_path), generator=generator) as engine:
        result = engine.analyze("Sum sales and plot by category", df)
    assert result.success and result.pass_at_1 and result.retry_count == 0
    assert result.provider == "deepseek" and result.model == "deepseek-flash"
    assert "Total: 70" in result.stdout
    assert Path(result.artifacts[0]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    record = json.loads((tmp_path / result.run_id / "result.json").read_text(encoding="utf-8"))
    assert record["pass_at_1"] and record["attempts"][0]["success"]
    assert "sample_rows" not in json.loads(generator.messages[0][1]["content"])["dataset"]


def test_error_feedback_and_fresh_dataframe(tmp_path, df):
    generator = ScriptedGenerator(
        "df['sales'] = 0\nraise ValueError('repair this')",
        "print(df['sales'].sum())",
    )
    engine = AnalysisEngine(Settings(output_dir=tmp_path), generator=generator)
    result = engine.analyze("Sum sales", df)
    assert result.success and not result.pass_at_1 and result.retry_count == 1
    assert result.stdout.strip() == "70"
    feedback = generator.messages[1][-1]
    assert feedback["role"] == "tool" and feedback["tool_call_id"] == "call-1"
    assert "repair this" in feedback["content"]
    assert df["sales"].tolist() == [10, 20, 40]


def test_correction_limit_and_no_failed_charts(tmp_path, df):
    code = "plt.plot([1, 2])\nplt.savefig('partial.png')\nraise ValueError('still wrong')"
    generator = ScriptedGenerator(code, code, code)
    result = AnalysisEngine(Settings(output_dir=tmp_path), generator=generator).analyze("plot", df)
    assert not result.success and len(result.attempts) == 3 and result.retry_count == 2
    assert result.artifacts == []
    assert all(attempt.artifacts == [] for attempt in result.attempts)


@pytest.mark.parametrize(
    "code",
    [
        "import os\nprint(os.environ)",
        "print(df.__class__)",
        "pd.read_csv('https://example.com/private.csv')",
        "df.to_csv('copy.csv')",
        "print(getattr(df, 'sales'))",
        "from pandas import read_csv\nprint(read_csv('https://example.com/data.csv'))",
        "from numpy import load\nprint(load('private.npy'))",
        "from matplotlib.pyplot import imread\nprint(imread('private.png'))",
    ],
)
def test_disallowed_operations(code):
    with pytest.raises(ValueError):
        validate_code(code)


def test_plot_path_restriction(tmp_path, df):
    result = PythonExecutor().execute("plt.savefig('../outside.png')", df, tmp_path / "attempt")
    assert not result.success and "plain PNG filename" in result.error
    assert not (tmp_path / "outside.png").exists()


def test_process_timeout(tmp_path, df):
    result = PythonExecutor(timeout_seconds=4).execute("while True: pass", df, tmp_path)
    assert not result.success and "timed out" in result.error


def test_worker_does_not_receive_api_key(tmp_path, df, monkeypatch):
    monkeypatch.setenv("API_KEY", "unit-test-secret-never-forward")
    import datatalk.executor as executor_module

    original_run = executor_module.subprocess.run

    def checked_run(*args, **kwargs):
        assert "API_KEY" not in kwargs["env"]
        return original_run(*args, **kwargs)

    monkeypatch.setattr(executor_module.subprocess, "run", checked_run)
    result = PythonExecutor().execute("print(df['sales'].sum())", df, tmp_path)
    assert result.success and "unit-test-secret" not in result.stdout


def test_empty_output_is_failure(tmp_path, df):
    result = PythonExecutor().execute("total = df['sales'].sum()", df, tmp_path)
    assert not result.success and "no printed answer" in result.error


def test_csv_input_limits_and_samples(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("category,sales\nA,10\nB,20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row limit"):
        load_csv(path, Settings(max_rows=1))
    frame = load_csv(path, Settings())
    assert "sample_rows" not in describe_dataframe(frame)
    assert describe_dataframe(frame, 1)["sample_rows"] == [{"category": "A", "sales": 10}]


def test_environment_precedence_and_missing_key(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("API_KEY=file-key\nAPI_MODEL=file-model\n", encoding="utf-8")
    monkeypatch.setenv("API_MODEL", "process-model")
    monkeypatch.delenv("API_KEY", raising=False)
    settings = Settings.from_env(path)
    assert settings.model == "process-model" and settings.api_key == "file-key"
    assert "API_KEY" not in __import__("os").environ
    with pytest.raises(ValueError, match="API_KEY"):
        OpenAICompatibleGenerator(Settings())


@pytest.mark.parametrize(
    ("provider", "base_url", "model"),
    [
        ("deepseek", "https://api.deepseek.com", "deepseek-flash"),
        ("openai", "https://api.openai.com/v1", "gpt-4.1-mini"),
        ("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
        ("moonshot", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
        ("zhipu", "https://open.bigmodel.cn/api/paas/v4", "glm-4.5"),
        ("siliconflow", "https://api.siliconflow.cn/v1", "Qwen/Qwen3-8B"),
        ("openrouter", "https://openrouter.ai/api/v1", "openai/gpt-4.1-mini"),
        (
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "gemini-2.5-flash",
        ),
    ],
)
def test_provider_presets(tmp_path, monkeypatch, provider, base_url, model):
    path = tmp_path / ".env"
    path.write_text(f"API_PROVIDER={provider}\nAPI_KEY=test\n", encoding="utf-8")
    for name in ("API_PROVIDER", "API_KEY", "API_BASE_URL", "API_MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env(path)
    assert settings.provider == provider
    assert settings.base_url == base_url
    assert settings.model == model


def test_custom_provider_and_catalog(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "API_PROVIDER=custom\nAPI_KEY=test\n"
        "API_BASE_URL=https://llm.example.com/v1\nAPI_MODEL=tool-model\n",
        encoding="utf-8",
    )
    for name in ("API_PROVIDER", "API_KEY", "API_BASE_URL", "API_MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env(path)
    assert settings.provider == "custom" and settings.model == "tool-model"
    assert {item["id"] for item in provider_catalog()} >= {"deepseek", "openai", "custom"}


def test_custom_provider_requires_endpoint_and_model(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("API_PROVIDER=custom\nAPI_KEY=test\n", encoding="utf-8")
    for name in ("API_PROVIDER", "API_KEY", "API_BASE_URL", "API_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="API_BASE_URL and API_MODEL"):
        Settings.from_env(path)


def test_api_adapter_request_and_malformed_responses():
    generator = OpenAICompatibleGenerator(Settings(api_key="unit-test-only"))
    generator.client.close()
    create = Mock()
    generator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    call = SimpleNamespace(
        id="call-unit",
        function=SimpleNamespace(
            name="execute_python",
            arguments=json.dumps({"code": "print(df.shape)"}),
        ),
    )
    choice = SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(
            content=None,
            tool_calls=[call],
        ),
    )
    create.return_value = SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=8,
        ),
    )
    generation = generator.generate([{"role": "user", "content": "shape"}])
    assert generation.code == "print(df.shape)" and generation.prompt_tokens == 12
    request = create.call_args.kwargs
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert request["tool_choice"]["function"]["name"] == "execute_python"
    choice.finish_reason = "length"
    choice.message.tool_calls = []
    with pytest.raises(ModelError, match="Expected an execute_python tool call"):
        generator.generate([])
    choice.finish_reason = "tool_calls"
    choice.message.tool_calls = [call]
    call.function.arguments = '{"code":'
    with pytest.raises(ModelError, match="invalid tool arguments"):
        generator.generate([])


def test_non_deepseek_request_omits_vendor_specific_fields():
    generator = OpenAICompatibleGenerator(
        Settings(
            provider="openai",
            api_key="unit-test-only",
            base_url="https://api.openai.com/v1",
            model="gpt-4.1-mini",
        )
    )
    generator.client.close()
    create = Mock()
    generator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    call = SimpleNamespace(
        id="call-openai",
        function=SimpleNamespace(
            name="execute_python", arguments=json.dumps({"code": "print(df.shape)"})
        ),
    )
    create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(content=None, tool_calls=[call]),
            )
        ],
        usage=None,
    )
    generator.generate([])
    assert "extra_body" not in create.call_args.kwargs


def test_provider_failure_does_not_retry(tmp_path, df):
    generator = Mock()
    generator.generate.side_effect = ModelError("OpenAI HTTP 401: check API_KEY")
    result = AnalysisEngine(Settings(output_dir=tmp_path), generator=generator).analyze("sum", df)
    assert not result.success and len(result.attempts) == 1
    assert generator.generate.call_count == 1 and "401" in result.error


@pytest.mark.parametrize("status_code", [200, 401])
def test_openai_sdk_http_request_without_network(status_code):
    requests = []

    def handle(request):
        requests.append(request)
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert request.headers["authorization"] == "Bearer unit-test-only"
        body = json.loads(request.content)
        assert body["thinking"] == {"type": "disabled"}
        assert body["tool_choice"]["function"]["name"] == "execute_python"
        if status_code == 401:
            return httpx.Response(401, json={"error": {"message": "invalid credentials"}})
        return httpx.Response(
            200,
            json={
                "id": "unit-response",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-http",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_python",
                                        "arguments": '{"code": "print(70)"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            },
        )

    generator = OpenAICompatibleGenerator(Settings(api_key="unit-test-only"))
    generator.client.close()
    generator.client = OpenAI(
        api_key="unit-test-only",
        base_url="https://api.deepseek.com",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    try:
        if status_code == 200:
            assert generator.generate([]).code == "print(70)"
        else:
            with pytest.raises(ModelError, match="HTTP 401") as caught:
                generator.generate([])
            assert "unit-test-only" not in str(caught.value)
        assert len(requests) == 1
    finally:
        generator.close()


@pytest.mark.parametrize("question", ["", " " * 3, "x" * 8001])
def test_invalid_questions_do_not_call_model(tmp_path, df, question):
    generator = Mock()
    engine = AnalysisEngine(Settings(output_dir=tmp_path), generator=generator)
    with pytest.raises(ValueError):
        engine.analyze(question, df)
    generator.generate.assert_not_called()


def test_run_directories_are_unique(tmp_path, df):
    generator = ScriptedGenerator("print(1)", "print(2)")
    engine = AnalysisEngine(Settings(output_dir=tmp_path), generator=generator)
    first = engine.analyze("first", df)
    second = engine.analyze("second", df)
    assert first.run_id != second.run_id and first.stdout.strip() == "1"
    assert second.stdout.strip() == "2"

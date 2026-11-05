"""Natural language -> tool call -> Python execution -> bounded correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .config import Settings
from .dataset import describe_dataframe, load_csv, validate_dataframe
from .executor import PythonExecutor
from .llm import CodeGenerator, ModelError, OpenAICompatibleGenerator
from .models import AnalysisResult, Attempt
from .prompts import SYSTEM_PROMPT


class AnalysisEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        generator: CodeGenerator | None = None,
    ) -> None:
        self.settings = settings if settings is not None else Settings.from_env()
        self._generator = generator
        self._owns_generator = generator is None
        self.executor = PythonExecutor(self.settings.execution_timeout_seconds)

    def __enter__(self) -> AnalysisEngine:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_generator and isinstance(self._generator, OpenAICompatibleGenerator):
            self._generator.close()
            self._generator = None

    def analyze_csv(
        self,
        question: str,
        csv_path: str | Path,
        *,
        encoding: str = "utf-8-sig",
    ) -> AnalysisResult:
        return self.analyze(question, load_csv(csv_path, self.settings, encoding))

    def analyze(self, question: str, df: pd.DataFrame) -> AnalysisResult:
        if not isinstance(question, str) or not question.strip() or len(question) > 8000:
            raise ValueError("Question must contain between 1 and 8000 characters")
        validate_dataframe(df, self.settings)
        metadata = describe_dataframe(df, self.settings.sample_rows)
        if self._generator is None:
            self._generator = OpenAICompatibleGenerator(self.settings)
        run_id = uuid4().hex
        run_dir = self.settings.output_dir.resolve() / run_id
        run_dir.mkdir(parents=True)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question.strip(), "dataset": metadata},
                    ensure_ascii=False,
                ),
            },
        ]
        result = AnalysisResult(
            success=False,
            question=question.strip(),
            run_id=run_id,
            provider=self.settings.provider,
            model=self.settings.model,
        )
        for number in range(1, self.settings.max_retries + 2):
            try:
                generation = self._generator.generate(messages)
            except ModelError as exc:
                # Transport/authentication/protocol errors stop without hidden API retries.
                result.error = str(exc)
                result.attempts.append(
                    Attempt(number=number, code="", success=False, error=str(exc))
                )
                break
            execution = self.executor.execute(
                generation.code,
                df,
                run_dir / f"attempt-{number}",
            )
            attempt = Attempt(
                number=number,
                code=generation.code,
                success=execution.success,
                stdout=execution.stdout,
                error=execution.error,
                artifacts=execution.artifacts,
                duration_seconds=execution.duration_seconds,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
            )
            result.attempts.append(attempt)
            result.code = generation.code
            result.stdout = execution.stdout
            result.error = execution.error
            if execution.success:
                result.success = True
                result.artifacts = execution.artifacts
                break
            messages.extend(
                [
                    generation.assistant_message,
                    {
                        "role": "tool",
                        "tool_call_id": generation.call_id,
                        "content": json.dumps(
                            {
                                "success": False,
                                "error": execution.error,
                                "instruction": "Fix code using df; call execute_python.",
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
            )
        (run_dir / "result.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

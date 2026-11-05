"""Frontend integration using the synchronous Python interface."""

from pathlib import Path

import pandas as pd

from datatalk import AnalysisEngine, Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = Settings.from_env(PROJECT_ROOT / ".env")
    df = pd.read_csv(PROJECT_ROOT / "examples" / "sales.csv")
    with AnalysisEngine(settings) as engine:
        result = engine.analyze("按 category 汇总 sales，打印结果并生成柱状图。", df)
    print(f"Provider: {result.provider} / {result.model}")
    print(result.stdout)
    print("Charts:", result.artifacts)
    if not result.success:
        print("Error:", result.error)


if __name__ == "__main__":
    main()

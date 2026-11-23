from pathlib import Path

import pandas as pd
import pytest

from datatalk.charts import english_column_labels
from datatalk.executor import PythonExecutor


@pytest.mark.parametrize(
    "label_code",
    [
        "plt.title('销售汇总')",
        "plt.xlabel('类别')",
        "plt.xticks([0, 1], ['食品', '服装'])",
        "plt.plot([1, 2], label='利润')\nplt.legend()",
        "plt.annotate('最高值', (0, 1))",
        "plt.table(cellText=[['销售额']], colLabels=['指标'], loc='bottom')",
        "image = plt.imshow([[1, 2], [3, 4]])\nplt.colorbar(image, label='数量')",
    ],
)
def test_worker_rejects_non_english_chart_text(tmp_path, label_code):
    code = f"plt.figure()\n{label_code}\nplt.savefig('output.png')"
    result = PythonExecutor().execute(code, pd.DataFrame({"sales": [1, 2]}), tmp_path)
    assert not result.success
    assert "Chart text must be in English" in result.error
    assert result.artifacts == []
    assert not (tmp_path / "output.png").exists()


def test_english_chart_keeps_chinese_calculation_and_printed_answer(tmp_path):
    data = pd.DataFrame({"销售额": [10, -2], "类别": ["食品", "服装"]})
    result = PythonExecutor().execute(
        "print('销售总额:', df['销售额'].sum())\n"
        "plt.bar(['Food', 'Clothing'], df['销售额'])\n"
        "plt.title('Sales by category')\nplt.xlabel('Category')\nplt.ylabel('Sales')\n"
        "plt.tight_layout()\nplt.savefig('sales.png')",
        data,
        tmp_path,
    )
    assert result.success, result.error
    assert "销售总额: 8" in result.stdout
    assert "Glyph" not in result.stdout
    assert Path(result.artifacts[0]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert list(data.columns) == ["销售额", "类别"]


def test_offline_labels_preserve_unknown_meaning_and_avoid_collisions():
    labels = english_column_labels(["销售额", "Sales", "自定义指标", "Column 3", "利润"])
    assert labels == ["Sales", "Sales (2)", "Column 3", "Column 3 (2)", "Profit"]

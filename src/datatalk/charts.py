"""Shared chart styling and English-label checks for both execution modes."""

from __future__ import annotations

import re
from collections.abc import Iterable

from matplotlib import cycler
from matplotlib.text import Text

# Match the workspace's Soft/indigo palette without requiring web fonts.
CHART_COLORS = ("#6366F1", "#14B8A6", "#A78BFA", "#F59E0B", "#38BDF8", "#F472B6")
CHART_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "text.color": "#1F2937",
    "figure.facecolor": "#FFFFFF",
    "figure.figsize": (9, 4.8),
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "savefig.facecolor": "#FFFFFF",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#E2E8F0",
    "axes.labelcolor": "#64748B",
    "axes.labelsize": 11,
    "axes.labelpad": 10,
    "axes.titlecolor": "#1F2937",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 20,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.prop_cycle": cycler(color=CHART_COLORS),
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.7,
    "xtick.color": "#64748B",
    "ytick.color": "#64748B",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 8,
    "ytick.major.pad": 8,
    "lines.linewidth": 2.4,
    "lines.markersize": 6,
    "patch.edgecolor": "#FFFFFF",
    "patch.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "legend.labelcolor": "#64748B",
    "image.cmap": "Purples",
}

_CJK_TEXT = re.compile("[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\U00020000-\U000323af]")
_COLUMN_NAMES = {
    "销售额": "Sales",
    "销售量": "Sales volume",
    "销量": "Sales volume",
    "利润": "Profit",
    "收入": "Revenue",
    "成本": "Cost",
    "价格": "Price",
    "单价": "Unit price",
    "数量": "Quantity",
    "金额": "Amount",
    "总额": "Total",
    "订单数": "Orders",
    "订单金额": "Order amount",
    "折扣": "Discount",
    "评分": "Rating",
    "年龄": "Age",
    "身高": "Height",
    "体重": "Weight",
    "温度": "Temperature",
    "湿度": "Humidity",
    "用电量": "Electricity consumption",
}


def english_column_labels(columns: Iterable[str]) -> list[str]:
    """Use known English names or explicit aliases; never guess unknown semantics."""
    labels: list[str] = []
    for index, column in enumerate(columns, start=1):
        original = str(column)
        label = _COLUMN_NAMES.get(original, original)
        if _CJK_TEXT.search(label):
            label = f"Column {index}"
        base = label
        suffix = 2
        while label in labels:
            label = f"{base} ({suffix})"
            suffix += 1
        labels.append(label)
    return labels


def validate_chart_text(figure) -> None:
    """Reject CJK labels so the normal model correction loop can translate them."""
    # Materialize automatic tick labels (including pandas categorical labels).
    texts = []
    for axis in figure.axes:
        axis.get_xticklabels()
        axis.get_yticklabels()
        # Matplotlib table cells do not expose their Text through findobj().
        for table in axis.tables:
            texts.extend(cell.get_text() for cell in table.get_celld().values())
    texts.extend(figure.findobj(match=Text))
    invalid = [
        artist.get_text()
        for artist in texts
        if artist.get_visible() and _CJK_TEXT.search(artist.get_text())
    ]
    if invalid:
        labels = ", ".join(repr(value[:80]) for value in dict.fromkeys(invalid))
        raise ValueError(
            "Chart text must be in English. Translate titles, axis labels, tick labels, "
            "legends, annotations and table headings before saving. "
            f"Labels requiring translation: {labels[:600]}"
        )

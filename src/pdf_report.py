"""将智能体求解过程编译为 PDF 报告（matplotlib + 系统中文字体）。"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages


def _find_cjk_font() -> str | None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def _wrap(text: str, width: int = 72) -> str:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return "\n".join(lines)


def _add_text_page(
    pdf: PdfPages,
    title: str,
    body: str,
    font_prop: font_manager.FontProperties | None,
):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.9])
    ax.axis("off")

    if font_prop:
        title_font = {"fontproperties": font_prop, "size": 16, "weight": "bold"}
        body_font = {"fontproperties": font_prop, "size": 10}
    else:
        title_font = {"size": 16, "weight": "bold"}
        body_font = {"size": 10}

    y = 0.98
    ax.text(0, y, title, transform=ax.transAxes, va="top", **title_font)
    y -= 0.04
    ax.text(0, y, _wrap(body, width=48), transform=ax.transAxes, va="top", **body_font)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf_report(
    output_path: Path,
    user_input: str,
    task_spec: dict[str, Any],
    recommendation: dict[str, Any],
    experiment: dict[str, Any] | None,
    tool_trace: list[dict[str, Any]],
    executive_summary: str,
    conclusions: str,
) -> Path:
    """生成完整求解过程 PDF。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_path = _find_cjk_font()
    font_prop = font_manager.FontProperties(fname=font_path) if font_path else None

    with PdfPages(str(output_path)) as pdf:
        cover = (
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "图机器学习问题优化与求解智能体\n"
            "DeepSeek API · 提示词工程 · 轻量工具调用"
        )
        _add_text_page(pdf, "图机器学习问题优化与求解报告", cover, font_prop)

        _add_text_page(pdf, "1. 用户原始问题", user_input, font_prop)
        _add_text_page(
            pdf,
            "2. 优化后问题",
            task_spec.get("optimized_question", "—"),
            font_prop,
        )
        _add_text_page(
            pdf,
            "3. 结构化任务规格",
            json.dumps(task_spec, ensure_ascii=False, indent=2),
            font_prop,
        )
        _add_text_page(
            pdf,
            "4. GNN 模型推荐",
            json.dumps(recommendation, ensure_ascii=False, indent=2),
            font_prop,
        )

        if experiment:
            exp_body = json.dumps(experiment, ensure_ascii=False, indent=2)
        else:
            exp_body = "未运行实验。"
        _add_text_page(pdf, "5. Baseline 实验结果", exp_body, font_prop)

        trace_lines = [
            f"[{i}] {step.get('tool', '?')}: {step.get('summary', '')}"
            for i, step in enumerate(tool_trace, 1)
        ]
        _add_text_page(
            pdf,
            "6. 工具调用轨迹",
            "\n".join(trace_lines) or "无",
            font_prop,
        )
        _add_text_page(pdf, "7. 执行摘要", executive_summary, font_prop)
        _add_text_page(pdf, "8. 结论与建议", conclusions, font_prop)

        d = pdf.infodict()
        d["Title"] = "GNN Agent Report"
        d["Author"] = "GNN Agent"
        d["CreationDate"] = datetime.now()

    return output_path

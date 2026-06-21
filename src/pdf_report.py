"""将智能体求解过程编译为 PDF 报告（reportlab 内置中文字体，兼容 Streamlit Cloud）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# 内置 CID 字体，无需系统/仓库外挂字体文件
_CJK_FONT = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(_CJK_FONT))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CJKTitle",
            parent=base["Heading1"],
            fontName=_CJK_FONT,
            fontSize=18,
            leading=24,
            spaceAfter=12,
        ),
        "heading": ParagraphStyle(
            "CJKHeading",
            parent=base["Heading2"],
            fontName=_CJK_FONT,
            fontSize=14,
            leading=20,
            spaceBefore=8,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "CJKBody",
            parent=base["Normal"],
            fontName=_CJK_FONT,
            fontSize=10,
            leading=15,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "mono": ParagraphStyle(
            "CJKMono",
            parent=base["Code"],
            fontName=_CJK_FONT,
            fontSize=8,
            leading=11,
            spaceAfter=4,
        ),
    }


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def _section(story: list, title: str, body: str, styles: dict[str, ParagraphStyle]):
    story.append(Paragraph(_escape(title), styles["heading"]))
    for line in body.split("\n"):
        if line.strip():
            story.append(Paragraph(_escape(line), styles["body"]))
        else:
            story.append(Spacer(1, 0.1 * cm))
    story.append(Spacer(1, 0.35 * cm))


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

    styles = _styles()
    story: list = []

    story.append(Paragraph("图机器学习问题优化与求解报告", styles["title"]))
    cover = (
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        "图机器学习问题优化与求解智能体<br/>"
        "DeepSeek API · 提示词工程 · 轻量工具调用"
    )
    story.append(Paragraph(cover, styles["body"]))
    story.append(Spacer(1, 0.5 * cm))

    _section(story, "1. 用户原始问题", user_input, styles)
    _section(
        story,
        "2. 优化后问题",
        task_spec.get("optimized_question", "—"),
        styles,
    )
    _section(
        story,
        "3. 结构化任务规格",
        json.dumps(task_spec, ensure_ascii=False, indent=2),
        styles,
    )
    _section(
        story,
        "4. GNN 模型推荐",
        json.dumps(recommendation, ensure_ascii=False, indent=2),
        styles,
    )

    if experiment:
        exp_body = json.dumps(experiment, ensure_ascii=False, indent=2)
    else:
        exp_body = "未运行实验。"
    _section(story, "5. Baseline 实验结果", exp_body, styles)

    trace_lines = [
        f"[{i}] {step.get('tool', '?')}: {step.get('summary', '')}"
        for i, step in enumerate(tool_trace, 1)
    ]
    _section(story, "6. 工具调用轨迹", "\n".join(trace_lines) or "无", styles)
    _section(story, "7. 执行摘要", executive_summary, styles)
    _section(story, "8. 结论与建议", conclusions, styles)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="GNN Agent Report",
        author="GNN Agent",
    )
    doc.build(story)
    return output_path

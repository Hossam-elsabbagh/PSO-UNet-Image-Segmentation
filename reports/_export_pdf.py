"""Export the editable Markdown report to a styled, reviewable PDF.

Run with the Codex primary runtime Python (which includes reportlab), or install
reportlab in another Python environment. This script does not invent results.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    LongTable,
    PageTemplate,
    Paragraph,
    Preformatted,
    Image as ReportImage,
    Spacer,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Phase1_Report.md"
OUTPUT = ROOT / "Phase1_Report.pdf"
PAGE_W, PAGE_H = A4
MARGIN_X = 48
MARGIN_TOP = 48
MARGIN_BOTTOM = 46
CONTENT_W = PAGE_W - 2 * MARGIN_X

font_sets = [
    (Path(r"C:\Windows\Fonts") / "arial.ttf",
     Path(r"C:\Windows\Fonts") / "arialbd.ttf",
     Path(r"C:\Windows\Fonts") / "consola.ttf"),
    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
     Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
     Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")),
    (Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf",
     Path(reportlab.__file__).resolve().parent / "fonts" / "VeraBd.ttf",
     None),
]
for regular_path, bold_path, mono_path in font_sets:
    if regular_path.is_file() and bold_path.is_file():
        pdfmetrics.registerFont(TTFont("Report", str(regular_path)))
        pdfmetrics.registerFont(TTFont("ReportBold", str(bold_path)))
        pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold")
        if mono_path is not None and mono_path.is_file():
            pdfmetrics.registerFont(TTFont("ReportMono", str(mono_path)))
            MONO_FONT = "ReportMono"
        else:
            MONO_FONT = "Courier"
        break
else:
    # ReportLab's base fonts keep regeneration possible without system TTFs.
    pdfmetrics.registerFontFamily("Helvetica", normal="Helvetica", bold="Helvetica-Bold")
    MONO_FONT = "Courier"

BODY_FONT = "Report" if regular_path.is_file() and bold_path.is_file() else "Helvetica"
BOLD_FONT = "ReportBold" if BODY_FONT == "Report" else "Helvetica-Bold"

INK = colors.HexColor("#1B2735")
MUTED = colors.HexColor("#536477")
BLUE = colors.HexColor("#135EA8")
PALE = colors.HexColor("#E9F2FB")
LINE = colors.HexColor("#CCD7E2")

styles = {
    "title": ParagraphStyle(
        "title", fontName=BOLD_FONT, fontSize=17.5, leading=21.5,
        textColor=INK, spaceAfter=10,
    ),
    "meta": ParagraphStyle(
        "meta", fontName=BODY_FONT, fontSize=8.4, leading=12,
        textColor=MUTED, spaceAfter=9,
    ),
    "h2": ParagraphStyle(
        "h2", fontName=BOLD_FONT, fontSize=11.4, leading=15,
        textColor=BLUE, spaceBefore=13, spaceAfter=6,
        keepWithNext=True,
    ),
    "body": ParagraphStyle(
        "body", fontName=BODY_FONT, fontSize=8.7, leading=12.5,
        textColor=INK, spaceAfter=7, allowWidows=0, allowOrphans=0,
        splitLongWords=True,
    ),
    "note": ParagraphStyle(
        "note", fontName=BODY_FONT, fontSize=8.3, leading=11.7,
        textColor=INK, leftIndent=10, rightIndent=10,
        borderColor=BLUE, borderWidth=0.6, borderPadding=8,
        backColor=PALE, spaceAfter=9,
    ),
    "table": ParagraphStyle(
        "table", fontName=BODY_FONT, fontSize=7.25, leading=9.8,
        textColor=INK, splitLongWords=True,
    ),
    "table_header": ParagraphStyle(
        "table_header", fontName=BOLD_FONT, fontSize=7.25,
        leading=9.8, textColor=INK, splitLongWords=True,
    ),
    "code": ParagraphStyle(
        "code", fontName=MONO_FONT, fontSize=7.4,
        leading=10.4, textColor=INK, leftIndent=8,
        backColor=colors.HexColor("#F2F5F8"), borderPadding=6,
        spaceAfter=8,
    ),
    "caption": ParagraphStyle(
        "caption", fontName=BODY_FONT, fontSize=7.5, leading=10.4,
        textColor=MUTED, alignment=TA_CENTER, spaceAfter=8,
    ),
}


def inline_markup(raw: str) -> str:
    """Translate a small, controlled subset of Markdown to ReportLab markup."""
    escaped = html.escape(raw, quote=False)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<link href="{html.escape(m.group(2), quote=True)}" color="#135EA8">{m.group(1)}</link>',
        escaped,
    )
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", lambda m: f'<font name="{MONO_FONT}" size="8">{m.group(1)}</font>', escaped)
    return escaped


class FlowDiagram(Flowable):
    """Compact flow diagram, with labels mirrored in the editable .mmd files."""

    def __init__(self, rows: list[list[str]], height: int = 74) -> None:
        super().__init__()
        self.rows = rows
        self.width = CONTENT_W
        self.height = height

    def draw(self) -> None:
        c = self.canv
        row_h = self.height / len(self.rows)
        for row_i, labels in enumerate(self.rows):
            n = len(labels)
            gap = 13
            box_w = (self.width - (n - 1) * gap) / n
            y = self.height - (row_i + 1) * row_h + 7
            box_h = row_h - 13
            for i, label in enumerate(labels):
                x = i * (box_w + gap)
                c.setFillColor(PALE if row_i % 2 == 0 else colors.white)
                c.setStrokeColor(LINE)
                c.roundRect(x, y, box_w, box_h, 4, stroke=1, fill=1)
                c.setFillColor(INK)
                c.setFont(BOLD_FONT, 7.1)
                c.drawCentredString(x + box_w / 2, y + box_h / 2 - 2.5, label)
                if i < n - 1:
                    end_x = x + box_w + gap - 3
                    mid_y = y + box_h / 2
                    c.setStrokeColor(BLUE)
                    c.line(x + box_w + 2, mid_y, end_x, mid_y)
                    c.line(end_x - 4, mid_y + 2.5, end_x, mid_y)
                    c.line(end_x - 4, mid_y - 2.5, end_x, mid_y)


def make_table(rows: list[list[str]]) -> LongTable:
    cols = len(rows[0])
    if cols == 4:
        widths = [87, 146, 147, CONTENT_W - 380]
    elif cols == 7:
        widths = [78, 61, 60, 51, 61, 71, CONTENT_W - 382]
    elif cols == 3:
        widths = [150, 135, CONTENT_W - 285]
    else:
        widths = [CONTENT_W / cols] * cols
    data = [
        [Paragraph(inline_markup(cell), styles["table_header"] if row_i == 0 else styles["table"])
         for cell in row]
        for row_i, row in enumerate(rows)
    ]
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, BLUE),
        ("GRID", (0, 1), (-1, -1), 0.25, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def diagrams() -> list:
    definitions = [
        ("Figure 1. Shared data pipeline and experiment branches.",
         [["Oxford-IIIT Pet", "Fixed split", "Trimap + valid mask"],
          ["Baseline + PSO branches", "Validation selection", "Held-out test"]]),
        ("Figure 2. Compact U-Net channel progression.",
         [["RGB 3", "Encoder 16/32/64/128", "Bottleneck 256", "Decoder 128/64/32/16", "1 logit"]]),
        ("Figure 3. PSO fitness feedback and final retraining.",
         [["Particle", "Fresh U-Net", "2-epoch train", "Validation Dice", "Best update"],
          ["Best parameters", "Fresh U-Net", "8-epoch final train", "Checkpoint", "Test later"]]),
        ("Figure 4. Preprocessing of each paired image and trimap.",
         [["RGB + trimap", "Resize", "Labels 1 / 2 / 3", "Target + valid mask", "Joint train flip"]]),
    ]
    output = []
    for caption, rows in definitions:
        output.append(KeepTogether([
            FlowDiagram(rows, 69 if len(rows) == 1 else 98),
            Paragraph(caption, styles["caption"]),
        ]))
    return output


def parse_markdown(source: str) -> list:
    lines = source.splitlines()
    story: list = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), styles["title"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), styles["h2"]))
            if line.startswith("## 7. "):
                story.extend(diagrams())
            i += 1
            continue
        if line.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(code_lines), styles["code"], maxLineLength=90))
            i += 1
            continue
        if line.startswith("!["):
            match = re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)", line)
            if match is None:
                raise ValueError(f"Malformed report image: {line}")
            caption, relative_path = match.groups()
            image_path = (ROOT / relative_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Report image is missing: {image_path}")
            illustration = ReportImage(str(image_path))
            scale = min(CONTENT_W / illustration.imageWidth,
                        360 / illustration.imageHeight, 1.0)
            illustration.drawWidth = illustration.imageWidth * scale
            illustration.drawHeight = illustration.imageHeight * scale
            story.append(KeepTogether([
                illustration,
                Paragraph(inline_markup(caption), styles["caption"]),
            ]))
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("|---"):
            table_lines = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table_lines.append([cell.strip() for cell in lines[i].strip().strip("|").split("|")])
                i += 1
            rows = [table_lines[0], *table_lines[2:]]
            story.extend([make_table(rows), Spacer(1, 7)])
            continue
        if line.startswith("> "):
            story.append(Paragraph(inline_markup(line[2:]), styles["note"]))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines):
            next_line = lines[i].strip()
            if (not next_line or next_line.startswith(("#", "```", "|", "> ", "!["))
                    or next_line == "---"):
                break
            paragraph.append(next_line)
            i += 1
        is_meta = len(story) == 1
        markup = ("<br/>".join(inline_markup(part) for part in paragraph)
                  if is_meta else inline_markup(" ".join(paragraph)))
        story.append(Paragraph(markup, styles["meta"] if is_meta else styles["body"]))
    return story


def decorate(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_X, PAGE_H - 31, PAGE_W - MARGIN_X, PAGE_H - 31)
    canvas.setFont(BODY_FONT, 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN_X, PAGE_H - 25, "DSAI 403  |  Project 08  |  Phase 1 draft")
    canvas.line(MARGIN_X, 33, PAGE_W - MARGIN_X, 33)
    canvas.drawString(MARGIN_X, 22, "Student review required  |  Results only from completed runs")
    canvas.drawRightString(PAGE_W - MARGIN_X, 22, str(doc.page))
    canvas.restoreState()


class ReportDoc(BaseDocTemplate):
    def __init__(self, output: Path) -> None:
        super().__init__(str(output), pagesize=A4, leftMargin=MARGIN_X,
                         rightMargin=MARGIN_X, topMargin=MARGIN_TOP,
                         bottomMargin=MARGIN_BOTTOM, title="Phase 1 Technical Report",
                         author="DSAI 403 Project 08 — draft for student review")
        frame = Frame(MARGIN_X, MARGIN_BOTTOM, CONTENT_W,
                      PAGE_H - MARGIN_TOP - MARGIN_BOTTOM, id="normal",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=decorate))


if __name__ == "__main__":
    report = ReportDoc(OUTPUT)
    source = SOURCE.read_text(encoding="utf-8")
    source = source.translate({ord(char): "-" for char in "\u2010\u2011\u2012\u2013\u2014"})
    report.build(parse_markdown(source))
    print(OUTPUT, OUTPUT.stat().st_size, "bytes")

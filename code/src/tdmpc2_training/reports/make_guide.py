"""Build the project guide PDF from a Markdown-like source (GUIDE.md).

From the repository root (reportlab is not in the project environment):

    ~/.local/bin/uv run --no-project --with reportlab python code/src/tdmpc2_training/reports/make_guide.py

Reads code/src/tdmpc2_training/reports/GUIDE.md and writes GUIDE.pdf next to
it. Supported source syntax: "# " chapter (starts a new page), "## " section,
"### " subsection, blank-line separated paragraphs, "- " bullet items (two
leading spaces for a second level), Markdown tables, ``` code blocks,
![caption](path from the repository root) figures, **bold**, *italic*
and `code` inside text.
"""
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FONTS = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DejaVu", str(FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONTS / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", str(FONTS / "DejaVuSansMono.ttf")))
addMapping("DejaVu", 0, 0, "DejaVu")
addMapping("DejaVu", 1, 0, "DejaVu-Bold")
addMapping("DejaVu", 0, 1, "DejaVu")          # no oblique face installed: italics stay upright
addMapping("DejaVu", 1, 1, "DejaVu-Bold")

INK = colors.HexColor("#1f2933")
ACCENT = colors.HexColor("#1f6f8b")
STYLES = {
    "Title": ParagraphStyle("Title", fontName="DejaVu-Bold", fontSize=24, leading=30, textColor=INK,
                            alignment=TA_CENTER, spaceAfter=10),
    "Subtitle": ParagraphStyle("Subtitle", fontName="DejaVu", fontSize=12, leading=17, textColor=INK,
                               alignment=TA_CENTER, spaceAfter=4),
    "H1": ParagraphStyle("H1", fontName="DejaVu-Bold", fontSize=18, leading=23, textColor=ACCENT,
                         spaceBefore=6, spaceAfter=12),
    "H2": ParagraphStyle("H2", fontName="DejaVu-Bold", fontSize=13.5, leading=18, textColor=INK,
                         spaceBefore=14, spaceAfter=6),
    "H3": ParagraphStyle("H3", fontName="DejaVu-Bold", fontSize=11, leading=15, textColor=INK,
                         spaceBefore=10, spaceAfter=4),
    "Body": ParagraphStyle("Body", fontName="DejaVu", fontSize=10, leading=14.5, textColor=INK, spaceAfter=7),
    "Bullet": ParagraphStyle("Bullet", fontName="DejaVu", fontSize=10, leading=14, textColor=INK,
                             leftIndent=16, bulletIndent=5, spaceAfter=3),
    "Bullet2": ParagraphStyle("Bullet2", fontName="DejaVu", fontSize=9.5, leading=13.5, textColor=INK,
                              leftIndent=32, bulletIndent=21, spaceAfter=2),
    "Cell": ParagraphStyle("Cell", fontName="DejaVu", fontSize=8, leading=10.5, textColor=INK),
    "CellHead": ParagraphStyle("CellHead", fontName="DejaVu-Bold", fontSize=8, leading=10.5, textColor=INK),
    "Code": ParagraphStyle("Code", fontName="DejaVuMono", fontSize=7.8, leading=10, textColor=INK,
                           backColor=colors.HexColor("#f2f4f6"), leftIndent=6, rightIndent=6,
                           borderPadding=5, spaceBefore=4, spaceAfter=9),
    "Caption": ParagraphStyle("Caption", fontName="DejaVu", fontSize=8.5, leading=11.5,
                              textColor=colors.HexColor("#52606d"), alignment=TA_CENTER, spaceAfter=12),
    "TOC1": ParagraphStyle("TOC1", fontName="DejaVu-Bold", fontSize=10.5, leading=15, leftIndent=0),
    "TOC2": ParagraphStyle("TOC2", fontName="DejaVu", fontSize=9.5, leading=13, leftIndent=16),
}


def inline(text):
    """Escape XML, then turn **bold**, *italic* and `code` into reportlab markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font name="DejaVuMono" size="8.6">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<i>\1</i>", text)
    return text


class GuideDoc(BaseDocTemplate):
    def __init__(self, path, title):
        super().__init__(str(path), pagesize=letter, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                         topMargin=0.8 * inch, bottomMargin=0.8 * inch, title=title,
                         author="Mounir Zouhari")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=self._footer)])
        self.heading_count = 0

    def beforeDocument(self):
        # multiBuild lays the document out several times until the contents
        # page settles; the bookmark names must be the same in every pass.
        self.heading_count = 0

    def _footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(colors.HexColor("#7b8794"))
        canvas.drawCentredString(letter[0] / 2, 0.45 * inch, f"{doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("H1", "H2"):
            level = 0 if flowable.style.name == "H1" else 1
            text = flowable.getPlainText()
            self.heading_count += 1
            key = f"heading{self.heading_count}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


def table_flowable(rows, width):
    header, body = rows[0], rows[1:]
    data = [[Paragraph(inline(c), STYLES["CellHead"]) for c in header]]
    data += [[Paragraph(inline(c), STYLES["Cell"]) for c in row] for row in body]
    columns = len(header)
    lengths = [max(len(r[i]) if i < len(r) else 0 for r in rows) + 4 for i in range(columns)]
    total = sum(lengths)
    widths = [max(0.7 * inch, width * n / total) for n in lengths]
    scale = width / sum(widths)
    table = Table(data, colWidths=[w * scale for w in widths], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4edf1")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7ced6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def figure(path, caption, width):
    full = REPO / path
    if not full.exists():
        return [Paragraph(f"[missing figure: {inline(path)}]", STYLES["Caption"])]
    w, h = ImageReader(str(full)).getSize()
    draw_w = min(width, 6.6 * inch)
    draw_h = draw_w * h / w
    if draw_h > 7.4 * inch:
        draw_h, draw_w = 7.4 * inch, 7.4 * inch * w / h
    return [KeepTogether([Image(str(full), width=draw_w, height=draw_h),
                          Paragraph(inline(caption), STYLES["Caption"])])]


def build(source, target):
    lines = source.read_text(encoding="utf-8").splitlines()
    doc = GuideDoc(target, title="TD-MPC2 Crazyflie project guide")
    width = doc.width
    story, paragraph, table, code, first_chapter = [], [], [], None, True

    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), STYLES["Body"]))
            paragraph.clear()

    def flush_table():
        if table:
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table]
            rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
            story.append(table_flowable(rows, width))
            story.append(Spacer(1, 9))
            table.clear()

    for line in lines:
        if code is not None:
            if line.startswith("```"):
                story.append(Preformatted("\n".join(code), STYLES["Code"]))
                code = None
            else:
                code.append(line)
            continue
        if line.startswith("```"):
            flush_paragraph(); flush_table()
            code = []
            continue
        if line.startswith("|"):
            flush_paragraph()
            table.append(line)
            continue
        flush_table()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        if line.startswith("% "):                      # title page lines
            flush_paragraph()
            style = "Title" if not any(isinstance(f, Paragraph) for f in story) else "Subtitle"
            story.append(Paragraph(inline(line[2:]), STYLES[style]))
            continue
        if line.startswith("[[TOC]]"):
            flush_paragraph()
            toc = TableOfContents()
            toc.levelStyles = [STYLES["TOC1"], STYLES["TOC2"]]
            story += [PageBreak(), Paragraph("Contents", STYLES["H1"]), toc]
            continue
        heading = re.match(r"(#{1,3}) (.*)", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            if level == 1:
                story.append(PageBreak())
                first_chapter = False
            story.append(Paragraph(inline(heading.group(2)), STYLES[f"H{level}"]))
            continue
        image = re.match(r"!\[(.*)\]\((.*)\)", stripped)
        if image:
            flush_paragraph()
            story += figure(image.group(2), image.group(1), width)
            continue
        bullet = re.match(r"( *)- (.*)", line)
        if bullet:
            flush_paragraph()
            style = "Bullet2" if len(bullet.group(1)) >= 2 else "Bullet"
            story.append(Paragraph(inline(bullet.group(2)), STYLES[style], bulletText="•"))
            continue
        if paragraph or not re.match(r"( *)- ", line):
            paragraph.append(stripped)
    flush_paragraph(); flush_table()
    doc.multiBuild(story)
    print(f"Wrote {target} ({doc.page} pages)")


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "GUIDE.md"
    build(source, source.with_suffix(".pdf"))

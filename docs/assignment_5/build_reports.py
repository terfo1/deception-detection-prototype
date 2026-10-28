"""Generate Assignment 5 PDFs with 12 pt Times and 1.5 line spacing.

Authoring dependency: reportlab (bundled Codex runtime); not needed by the app.
Explicit page markers make the required page counts reproducible.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args()
    for name, file in [("ReportTimes", "times.ttf"), ("ReportTimesBold", "timesbd.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(args.font_dir / file)))
    pdfmetrics.registerFontFamily("ReportTimes", normal="ReportTimes", bold="ReportTimesBold")
    body = ParagraphStyle("body", fontName="ReportTimes", fontSize=12, leading=18,
                          alignment=TA_JUSTIFY, spaceAfter=8)
    heading = ParagraphStyle("heading", parent=body, fontName="ReportTimesBold", alignment=TA_LEFT,
                             spaceBefore=4, spaceAfter=10, keepWithNext=True)
    caption = ParagraphStyle("caption", parent=body, alignment=TA_LEFT, spaceAfter=6)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("ReportTimes", 10)
        canvas.setFillColor(colors.HexColor("#465467"))
        canvas.drawString(50, 28, "Assignment 5 | Eye-tracking research prototype")
        canvas.drawRightString(A4[0] - 50, 28, str(document.page))
        canvas.restoreState()

    for source_name, output_name in [("THEORY_EN.md", "Assignment_5_Theory_EN.pdf"),
                                      ("PRACTICAL_EN.md", "Assignment_5_Practical_EN.pdf")]:
        source = Path(__file__).with_name(source_name)
        output = args.output_dir / output_name
        if output.exists():
            raise FileExistsError(output)
        story = []
        pages = source.read_text(encoding="utf-8").split("<!-- page -->")
        for page_index, page in enumerate(pages):
            if page_index:
                story.append(PageBreak())
            for block in page.strip().split("\n\n"):
                block = block.strip()
                image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", block)
                if image_match:
                    label, relative = image_match.groups()
                    img = Image(str(source.parent / relative))
                    scale = min((A4[0] - 100) / img.imageWidth, 205 / img.imageHeight)
                    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
                    story.extend([img, Spacer(1, 4), Paragraph(escape(label), caption)])
                elif block:
                    style = heading if block.startswith("#") else body
                    text = escape(re.sub(r"^#+\s*", "", block).replace("\n", " "))
                    text = re.sub(r"https://[^\s]+", lambda m: f'<link href="{m.group(0)}" color="#194b83">{m.group(0)}</link>', text)
                    story.append(Paragraph(text, style))
        doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=50, leftMargin=50,
                                topMargin=44, bottomMargin=48,
                                title=source_name.replace("_EN.md", "") + " - Assignment 5", author="")
        doc.build(story, onFirstPage=footer, onLaterPages=footer)
        print(f"Created {output}; planned pages: {len(pages)}")


if __name__ == "__main__":
    main()

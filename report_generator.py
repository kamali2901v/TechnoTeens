"""
Generates a professional, one-page PDF quality report for a batch.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
)
from datetime import datetime

from db import get_batch_summary, get_all_batches, get_batch_samples

BROWN = colors.HexColor("#8B4513")
LIGHT_TAN = colors.HexColor("#F5DEB3")
DARK_GREY = colors.HexColor("#222222")
MID_GREY = colors.HexColor("#555555")


def calculate_grade(healthy_pct):
    """
    Two-tier grading matching the official SIH problem statement:
    Grade A = high quality, meets full specification.
    URS = Under Relaxed Specifications (NAFED procurement category
    for smaller/slightly blemished onions still accepted for purchase).
    """
    if healthy_pct >= 80:
        return "GRADE A", colors.HexColor("#E6F4EA"), colors.HexColor("#1E7A34")
    else:
        return "URS (Under Relaxed Specifications)", colors.HexColor("#FFF4E0"), colors.HexColor("#B9770E")


def generate_batch_report(batch_id, output_path=None, officer_name=None, officer_id=None):
    if output_path is None:
        output_path = f"reports/{batch_id}_report.pdf"
    os.makedirs("reports", exist_ok=True)

    all_batches = get_all_batches()
    batch_info = next((b for b in all_batches if b["batch_id"] == batch_id), None)
    summary = get_batch_summary(batch_id)

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.5 * inch, bottomMargin=0.45 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    # --- Styles ---
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=19,
        textColor=BROWN, spaceAfter=1, alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"], fontSize=10.5,
        textColor=MID_GREY, spaceAfter=10, alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], fontSize=12,
        textColor=DARK_GREY, spaceBefore=10, spaceAfter=5
    )
    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["Normal"], fontSize=10.5, leading=15,
        textColor=DARK_GREY
    )
    note_style = ParagraphStyle(
        "NoteStyle", parent=styles["Normal"], fontSize=8.5,
        textColor=MID_GREY, leading=11.5
    )
    grade_style = ParagraphStyle(
        "GradeStyle", parent=styles["Normal"], fontSize=20,
        leading=24, alignment=TA_CENTER, fontName="Helvetica-Bold"
    )
    evidence_caption_style = ParagraphStyle(
        "EvidenceCaption", parent=styles["Normal"], fontSize=8,
        textColor=DARK_GREY, alignment=TA_CENTER, leading=10.5
    )

    # --- Header ---
    story.append(Paragraph(" AgroNex ", title_style))
    story.append(Paragraph("Onion Quality Assessment Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_TAN, spaceAfter=10))

    # --- Batch details ---
    if batch_info:
        batch_data = [
            ["Batch ID", batch_info["batch_id"], "Date", batch_info["date"] or "-"],
            ["Supplier", batch_info["supplier_name"] or "-", "Variety", batch_info["onion_variety"] or "-"],
            ["Centre", batch_info["procurement_centre"] or "-", "Quantity",
             f"{batch_info['quantity_received']} {batch_info['unit']}"],
            ["Inspector", officer_name or "-", "Officer ID", officer_id or "-"],
        ]
        batch_table = Table(batch_data, colWidths=[65, 175, 65, 175])
        batch_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), LIGHT_TAN),
            ("BACKGROUND", (2, 0), (2, -1), LIGHT_TAN),
            ("TEXTCOLOR", (0, 0), (0, -1), BROWN),
            ("TEXTCOLOR", (2, 0), (2, -1), BROWN),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9D9D9")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(batch_table)
        story.append(Spacer(1, 12))

    total = summary["total_samples"]
    healthy_pct = summary["healthy_pct"]

    # --- Quality breakdown ---
    story.append(Paragraph("Quality Breakdown", heading_style))
    quality_data = [
        ["Category", "Count", "Percentage"],
        ["Total Samples Inspected", str(total), "—"],
        ["Healthy", str(summary["healthy"]), f"{summary['healthy_pct']}%"],
        ["Damaged", str(summary["damaged"]), f"{summary['damaged_pct']}%"],
        ["Rotten", str(summary["rotten"]), f"{summary['rotten_pct']}%"],
    ]
    quality_table = Table(quality_data, colWidths=[240, 110, 130])
    quality_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BROWN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FAFAFA")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9D9D9")),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(quality_table)
    story.append(Spacer(1, 14))

    # --- FINAL GRADE (prominent banner) ---
    if total == 0:
        grade_text, grade_bg, grade_fg = "NOT YET GRADED", colors.HexColor("#F0F0F0"), MID_GREY
    else:
        grade_text, grade_bg, grade_fg = calculate_grade(healthy_pct)

    grade_label_style = ParagraphStyle(
        "GradeLabel", parent=styles["Normal"], fontSize=9,
        textColor=MID_GREY, alignment=TA_CENTER, spaceAfter=2
    )
    grade_value_style = ParagraphStyle(
        "GradeValue", parent=grade_style, textColor=grade_fg
    )
    grade_table = Table(
        [[Paragraph("FINAL GRADE", grade_label_style)],
         [Paragraph(grade_text, grade_value_style)]],
        colWidths=[480]
    )
    grade_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), grade_bg),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 1, grade_fg),
    ]))
    story.append(grade_table)
    story.append(Spacer(1, 14))

    # --- Size check ---
    

    # --- Sample Evidence (actual photos with AI result, as proof) ---
    samples = get_batch_samples(batch_id)
    if samples:
        story.append(Paragraph("Sample Evidence", heading_style))
        story.append(Paragraph(
            "Each sample below shows the actual photo captured during inspection, "
            "with the AI prediction and confidence recorded as supporting evidence.",
            note_style
        ))
        story.append(Spacer(1, 6))

        thumb_size = 1.3 * inch
        cell_width = 160
        row_cells = []
        grid_rows = []

        for sample in samples:
            final = (sample["human_decision"] or sample["ai_prediction"] or "").upper()
            conf_pct = f"{sample['ai_confidence']*100:.0f}%" if sample["ai_confidence"] is not None else "-"
            caption_text = f"{sample['sample_id']}<br/><b>{final}</b> ({conf_pct})"

            img_path = sample["image_path"]
            if img_path and os.path.isfile(img_path):
                try:
                    img_flowable = Image(img_path, width=thumb_size, height=thumb_size)
                except Exception:
                    img_flowable = Paragraph("(image unavailable)", evidence_caption_style)
            else:
                img_flowable = Paragraph("(image unavailable)", evidence_caption_style)

            cell = Table(
                [[img_flowable], [Paragraph(caption_text, evidence_caption_style)]],
                colWidths=[cell_width]
            )
            cell.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            row_cells.append(cell)

            if len(row_cells) == 3:
                grid_rows.append(row_cells)
                row_cells = []

        if row_cells:
            while len(row_cells) < 3:
                row_cells.append(Paragraph("", evidence_caption_style))
            grid_rows.append(row_cells)

        evidence_table = Table(grid_rows, colWidths=[cell_width, cell_width, cell_width])
        evidence_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(evidence_table)
        story.append(Spacer(1, 12))

    # --- Footer ---
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#D9D9D9"), spaceAfter=6))
    story.append(Paragraph(
        "Grading shown is provisional pending official AGMARK/NAFED verification. "
        "URS = Under Relaxed Specifications (NAFED procurement category below Grade A). ",
        note_style
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · AgroNex  AI",
        note_style
    ))

    doc.build(story)
    return output_path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from pathlib import Path

DATA = [
    ["Model", "Params (M)", "Inference (ms/img)", "Precision (P)", "Recall (R)", "mAP50", "mAP50-95"],
    ["yolov8n", "~3.2", "~118", "0.64", "0.537", "0.605", "0.446"],
    ["yolov8s", "~11.2", "~244", "0.797", "0.664", "0.760", "0.589"],
    ["yolov8m", "~25.9", "~482", "0.712", "0.730", "0.784", "0.614"],
]

def build_pdf(out_path: Path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, title="YOLO Comparison")
    elems = []
    elems.append(Paragraph("YOLOv8 Model Comparison (coco128 quick eval)", styles["Title"]))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph("Note: coco128 is a lightweight subset intended for quick comparisons, not final COCO accuracy.", styles["Normal"]))
    elems.append(Spacer(1, 12))
    t = Table(DATA, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#233055')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#e6eef8')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#233055')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f6f8fc')]),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('FONTSIZE', (0,1), (-1,-1), 11),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 18))
    elems.append(Paragraph("Why we used yolov8n in the MVP", styles["Heading2"]))
    elems.append(Paragraph("- Lowest latency and smallest footprint for real-time assistive guidance on commodity devices.", styles["Normal"]))
    elems.append(Paragraph("- Adequate detection quality when paired with BLIP captioning and our narrative reasoner.", styles["Normal"]))
    elems.append(Paragraph("- Larger models (s/m) yield higher mAP but increase latency and power draw; they remain configurable via YOLO_VARIANT.", styles["Normal"]))
    doc.build(elems)

if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'run' / 'YOLO_comparison.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(out)
    print(f"Wrote {out}")


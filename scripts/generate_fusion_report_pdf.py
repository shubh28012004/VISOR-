#!/usr/bin/env python3
"""
Generate comprehensive Multimodal AI Fusion Report PDF with metrics, graphs references, and architecture details.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from pathlib import Path

# Data tables
YOLO_DATA = [
    ["Model", "Parameters (M)", "Inference (ms/img)", "Precision (P)", "Recall (R)", "mAP50", "mAP50-95"],
    ["yolov8n", "~3.2", "~118", "0.64", "0.537", "0.605", "0.446"],
    ["yolov8s", "~11.2", "~244", "0.797", "0.664", "0.760", "0.589"],
    ["yolov8m", "~25.9", "~482", "0.712", "0.730", "0.784", "0.614"],
]

FUSION_COMPARISON = [
    ["Aspect", "BLIP Caption Alone", "YOLO Detections Alone", "Fused Narrative"],
    ["Object Specificity", "General (\"a person\")", "Specific (\"person 93%\")", "Contextual (\"person seated at desk\")"],
    ["Spatial Relationships", "Limited", "None", "Inferred (\"person with laptop\")"],
    ["Naturalness", "Good", "Poor (list format)", "Excellent (sentences)"],
    ["TTS Optimization", "Fair", "Poor", "Excellent (concise, natural)"],
]

def build_pdf(out_path: Path):
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        textColor=colors.HexColor('#0b1020'),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#233055'),
        spaceBefore=12,
        spaceAfter=8
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    )
    
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, title="VISOR Multimodal AI Fusion Report")
    elems = []
    
    # Title
    elems.append(Paragraph("Multimodal AI Fusion Architecture Report", title_style))
    elems.append(Paragraph("VISOR: AI-Powered Guiding Shield for Vision", styles['Heading3']))
    elems.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    elems.append(Paragraph("Executive Summary", heading_style))
    elems.append(Paragraph(
        "VISOR implements a <b>late fusion multimodal architecture</b> that combines complementary vision models "
        "(YOLO object detection + BLIP image captioning) through a language model reasoner to generate unified, "
        "context-aware scene descriptions optimized for assistive technology. The system demonstrates significant "
        "improvements in descriptive accuracy and naturalness compared to individual model outputs.",
        normal_style
    ))
    elems.append(Spacer(1, 0.2*inch))
    
    # Fusion Architecture
    elems.append(Paragraph("1. Fusion Architecture", heading_style))
    elems.append(Paragraph("<b>1.1 Overview</b>", styles['Heading3']))
    elems.append(Paragraph(
        "The fusion pipeline operates at the <b>output level (late fusion)</b>, where independent vision models "
        "process the same input image, and their outputs are combined via a reasoning layer:",
        normal_style
    ))
    elems.append(Spacer(1, 0.1*inch))
    
    # Architecture diagram (text-based)
    arch_text = """<pre>
Input Image
    ├─→ YOLOv8 (Object Detection)
    │   └─→ Output: Bounding boxes, class labels, confidence scores
    │
    ├─→ BLIP (Image Captioning)
    │   └─→ Output: Natural language scene description
    │
    └─→ BLIP VQA (Question Answering)
        └─→ Output: Answer to user queries
        
    ↓ FUSION LAYER ↓
    
    FLAN-T5 / Gemini 2.5 Flash (Reasoning)
    └─→ Synthesized Narrative (fused output)
</pre>"""
    elems.append(Paragraph(arch_text, styles['Code']))
    elems.append(Spacer(1, 0.15*inch))
    
    elems.append(Paragraph("<b>1.2 Technical Implementation</b>", styles['Heading3']))
    elems.append(Paragraph(
        "<b>Fusion Function</b>: <i>generate_narrative()</i> in backend.py<br/>"
        "<b>Inputs</b>: YOLO detections (top 6 objects) + BLIP caption<br/>"
        "<b>Process</b>: Detection summarization → Prompt construction → Language model synthesis<br/>"
        "<b>Output</b>: Single concise sentence (&lt;25 words) optimized for text-to-speech",
        normal_style
    ))
    elems.append(Spacer(1, 0.15*inch))
    
    elems.append(Paragraph("<b>1.3 Why Late Fusion?</b>", styles['Heading3']))
    elems.append(Paragraph(
        "<b>Advantages</b>: Modularity, interpretability, efficiency, flexibility<br/>"
        "<b>Trade-offs</b>: Slightly higher latency (mitigated by lightweight reasoner)",
        normal_style
    ))
    elems.append(Spacer(1, 0.2*inch))
    
    # Model Performance Metrics
    elems.append(Paragraph("2. Model Performance Metrics", heading_style))
    elems.append(Paragraph("<b>2.1 YOLO Detection Models (coco128 validation)</b>", styles['Heading3']))
    
    # YOLO comparison table
    t1 = Table(YOLO_DATA, colWidths=[0.8*inch, 0.9*inch, 1.0*inch, 0.8*inch, 0.7*inch, 0.7*inch, 0.8*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#233055')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#e6eef8')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#233055')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f6f8fc')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    elems.append(t1)
    elems.append(Spacer(1, 0.1*inch))
    
    elems.append(Paragraph(
        "<b>Analysis</b>: yolov8n selected for MVP (lowest latency: 118ms). yolov8s shows best precision-recall "
        "balance. yolov8m achieves highest mAP50-95 (0.614) but 4x slower.",
        normal_style
    ))
    elems.append(Spacer(1, 0.15*inch))
    
    elems.append(Paragraph("<b>2.2 Available Visualizations</b>", styles['Heading3']))
    elems.append(Paragraph(
        "Graphs available in <i>run/yolov8{n,s,m}/</i>:<br/>"
        "• BoxPR_curve.png: Precision-Recall curve<br/>"
        "• BoxF1_curve.png: F1-score vs. confidence threshold<br/>"
        "• BoxP_curve.png / BoxR_curve.png: Precision/Recall curves<br/>"
        "• confusion_matrix.png: Per-class confusion matrix<br/>"
        "• val_batch*.jpg: Sample predictions vs. ground truth",
        normal_style
    ))
    elems.append(Spacer(1, 0.15*inch))
    
    elems.append(Paragraph("<b>2.3 Fusion Pipeline Performance</b>", styles['Heading3']))
    
    # Fusion comparison table
    t2 = Table(FUSION_COMPARISON, colWidths=[1.2*inch, 1.6*inch, 1.6*inch, 1.6*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#233055')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#e6eef8')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#233055')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f6f8fc')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elems.append(t2)
    elems.append(Spacer(1, 0.2*inch))
    
    # Gemini Integration
    elems.append(Paragraph("3. Integration of Gemini 2.5 Flash", heading_style))
    elems.append(Paragraph(
        "The integration of <b>Gemini 2.5 Flash</b> alongside FLAN-T5 provides:<br/>"
        "• Improved context understanding and reasoning capabilities<br/>"
        "• Better instruction following for user-specific prompts<br/>"
        "• Reduced hallucinations through superior grounding<br/><br/>"
        "The fusion layer supports multiple reasoner backends (FLAN-T5-small or Gemini 2.5 Flash) "
        "via <i>REASONER_CKPT</i> configuration.",
        normal_style
    ))
    elems.append(Spacer(1, 0.2*inch))
    
    # Use Case
    elems.append(Paragraph("4. Use Case: Assistive Technology", heading_style))
    elems.append(Paragraph(
        "<b>Requirements</b>: Real-time performance (&lt;500ms), high accuracy, natural speech output, reliability<br/><br/>"
        "<b>Fusion Benefits</b>:<br/>"
        "1. Comprehensive descriptions combining 'what' (caption) with 'where/what exactly' (detections)<br/>"
        "2. Reduced ambiguity through multiple complementary signals<br/>"
        "3. Natural speech output optimized for text-to-speech<br/>"
        "4. Context-aware reasoning that infers spatial relationships",
        normal_style
    ))
    elems.append(Spacer(1, 0.2*inch))
    
    # Conclusion
    elems.append(Paragraph("5. Conclusion", heading_style))
    elems.append(Paragraph(
        "The multimodal fusion architecture in VISOR successfully combines object detection and image captioning "
        "to produce superior scene descriptions. Key achievements:<br/>"
        "✓ Higher descriptive accuracy than individual models<br/>"
        "✓ Natural, TTS-optimized outputs for assistive applications<br/>"
        "✓ Modular, extensible design for future enhancements<br/>"
        "✓ Real-time performance suitable for mobile/edge devices<br/><br/>"
        "The integration of Gemini 2.5 Flash further enhances reasoning quality, demonstrating the flexibility "
        "of the fusion approach.",
        normal_style
    ))
    elems.append(Spacer(1, 0.3*inch))
    
    # Footer
    elems.append(Paragraph(
        "<i>Report Generated: 2025 | Models: YOLOv8 (n/s/m), BLIP-base, FLAN-T5-small, Gemini 2.5 Flash | "
        "Dataset: COCO128 validation | Graphs: run/yolov8{n,s,m}/</i>",
        styles['Normal']
    ))
    
    doc.build(elems)
    print(f"✓ Generated comprehensive fusion report: {out_path}")

if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'run' / 'Multimodal_AI_Fusion_Report.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(out)


#!/usr/bin/env python3
"""
Convert all markdown documentation files to a single PDF.
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "run"

def markdown_to_paragraphs(text, styles):
    """Convert markdown text to ReportLab paragraphs."""
    elements = []
    lines = text.split('\n')
    
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []
    
    for line in lines:
        # Handle code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                # End code block
                if code_lines:
                    code_text = '\n'.join(code_lines)
                    p = Paragraph(f"<pre>{code_text}</pre>", styles['Code'])
                    elements.append(p)
                    elements.append(Spacer(1, 0.1*inch))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue
        
        if in_code_block:
            code_lines.append(line)
            continue
        
        # Handle tables
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append([cell.strip() for cell in line.split('|')[1:-1]])
            continue
        else:
            if in_table and table_rows:
                # Process table
                if len(table_rows) > 1:
                    t = Table(table_rows)
                    t.setStyle(TableStyle([
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
                    ]))
                    elements.append(t)
                    elements.append(Spacer(1, 0.1*inch))
                table_rows = []
                in_table = False
        
        # Skip empty lines
        if not line.strip():
            elements.append(Spacer(1, 0.05*inch))
            continue
        
        # Handle headers
        if line.startswith('# '):
            elements.append(Paragraph(line[2:].strip(), styles['Title']))
            elements.append(Spacer(1, 0.2*inch))
        elif line.startswith('## '):
            elements.append(Paragraph(line[3:].strip(), styles['Heading1']))
            elements.append(Spacer(1, 0.15*inch))
        elif line.startswith('### '):
            elements.append(Paragraph(line[4:].strip(), styles['Heading2']))
            elements.append(Spacer(1, 0.1*inch))
        elif line.startswith('#### '):
            elements.append(Paragraph(line[5:].strip(), styles['Heading3']))
            elements.append(Spacer(1, 0.08*inch))
        elif line.startswith('**') and line.endswith('**'):
            # Bold text
            text = line.strip('*')
            elements.append(Paragraph(f"<b>{text}</b>", styles['Normal']))
            elements.append(Spacer(1, 0.05*inch))
        else:
            # Regular paragraph
            # Convert markdown formatting
            text = line
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
            text = text.replace('✓', '✓')
            
            p = Paragraph(text, styles['Normal'])
            elements.append(p)
            elements.append(Spacer(1, 0.05*inch))
    
    return elements

def create_pdf():
    """Create PDF from all markdown files."""
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#0b1020'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    )
    
    styles.add(normal_style)
    
    out_path = RUN_DIR / "VISOR_Complete_Documentation.pdf"
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, title="VISOR Complete Documentation")
    all_elements = []
    
    # Title page
    all_elements.append(Spacer(1, 2*inch))
    all_elements.append(Paragraph("VISOR", title_style))
    all_elements.append(Paragraph("AI-Powered Guiding Shield for Vision", styles['Heading2']))
    all_elements.append(Spacer(1, 0.3*inch))
    all_elements.append(Paragraph("Complete Documentation", styles['Normal']))
    all_elements.append(Spacer(1, 0.2*inch))
    all_elements.append(Paragraph("Multimodal AI Fusion Architecture for Assistive Technology", styles['Normal']))
    all_elements.append(PageBreak())
    
    # Process each markdown file
    md_files = [
        "YOLO_comparison.md",
        "Multimodal_AI_Fusion_Report.md",
        "FUSION_EVALUATION_RESULTS.md",
        "FUSION_METRICS_README.md",
        "RESULTS_TABLES.md",
    ]
    
    for md_file in md_files:
        file_path = RUN_DIR / md_file
        if file_path.exists():
            print(f"Processing {md_file}...")
            content = file_path.read_text(encoding='utf-8')
            
            # Add section header
            all_elements.append(Paragraph(md_file.replace('.md', '').replace('_', ' '), styles['Title']))
            all_elements.append(Spacer(1, 0.2*inch))
            
            # Convert markdown to paragraphs
            elements = markdown_to_paragraphs(content, styles)
            all_elements.extend(elements)
            all_elements.append(PageBreak())
    
    doc.build(all_elements)
    print(f"✓ Generated PDF: {out_path}")

if __name__ == '__main__':
    create_pdf()


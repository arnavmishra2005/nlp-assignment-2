"""Render the supplied Q4 technical report and labelled proxy-analysis appendix."""
from pathlib import Path
import html,re
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent

def build_report():
    font=Path('C:/Windows/Fonts/arial.ttf'); normal='Helvetica';bold='Helvetica-Bold'
    if font.exists():
        pdfmetrics.registerFont(TTFont('Q4Arial',str(font)))
        pdfmetrics.registerFont(TTFont('Q4ArialBold',str(font.with_name('arialbd.ttf'))))
        normal='Q4Arial';bold='Q4ArialBold'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Q4Body',fontName=normal,fontSize=9,leading=12,spaceAfter=7))
    styles.add(ParagraphStyle(name='Q4Cell',fontName=normal,fontSize=7.2,leading=9.5,spaceAfter=0))
    styles.add(ParagraphStyle(name='Q4Title',fontName=bold,fontSize=18,leading=22,spaceAfter=12))
    styles.add(ParagraphStyle(name='Q4Heading',fontName=bold,fontSize=12,leading=15,spaceBefore=8,spaceAfter=7,keepWithNext=True))
    def markup(text):
        text=re.sub(r'\\([<>*_])',r'\1',text)
        text=html.escape(text)
        text=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)
        text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
        text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'\1',text)
        return text
    story=[]
    def render(path):
        lines=path.read_text(encoding='utf-8-sig').splitlines();i=0
        while i<len(lines):
            line=lines[i].strip()
            if not line:i+=1;continue
            if line.startswith('```'):
                i+=1;code=[]
                while i<len(lines) and not lines[i].strip().startswith('```'):code.append(lines[i]);i+=1
                for text in code:story.append(Paragraph(html.escape(text) or ' ',styles['Q4Cell']))
                story.append(Spacer(1,8));i+=1;continue
            if re.match(r'^\|',line):
                data=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                    if not all(re.fullmatch(r':?-+:?',x or ' ') for x in row):data.append(row)
                    i+=1
                width=max(len(x) for x in data)
                values=[[Paragraph(markup(x),styles['Q4Cell']) for x in row+['']*(width-len(row))] for row in data]
                widths=[490/width]*width
                if width==4 and any('Output' in x for x in data[0]):widths=[72,98,235,85]
                t=Table(values,colWidths=widths,repeatRows=1,splitInRow=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.4,colors.grey),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
                story.extend([t,Spacer(1,8)]);continue
            image=re.match(r'^!\[([^]]*)\]\(([^)]+)\)',line)
            if image:
                target=(ROOT/image.group(2)).resolve()
                if not target.is_relative_to(ROOT.resolve()):raise ValueError('Image path escapes Q4')
                im=Image(str(target));scale=min(490/im.imageWidth,230/im.imageHeight);im.drawWidth=scale*im.imageWidth;im.drawHeight=scale*im.imageHeight
                story.extend([im,Paragraph(markup(image.group(1)),styles['Q4Cell']),Spacer(1,8)]);i+=1;continue
            if line.startswith('#'):
                heading=line.lstrip('#').strip();style=styles['Q4Title'] if line.startswith('# ') else styles['Q4Heading']
                story.append(Paragraph(markup(heading),style));i+=1;continue
            if line in ['---','***']:story.append(Spacer(1,8));i+=1;continue
            if re.match(r'^[-*] ',line):
                story.append(Paragraph('&#8226; '+markup(line[2:]),styles['Q4Body']));i+=1;continue
            paragraph=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','|','```','- ','* ','![')):
                paragraph.append(lines[i].strip());i+=1
            story.append(Paragraph(markup(' '.join(paragraph)),styles['Q4Body']))
    render(ROOT/'Plot-Twist Generator RNN vs LSTM.md')
    story.append(PageBreak())
    story.append(Paragraph('Appendix: AI-simulated proxy evaluation',styles['Q4Title']))
    render(ROOT/'evaluation/AI_Simulated_Evaluation_Analysis.md')
    output=ROOT/'Plot_Twist_Generator_Final_Report.pdf'
    def footer(c,doc):
        c.setFont('Helvetica',8);c.drawString(42,22,'DSL 504 | Question 4 | Plot-Twist Generator');c.drawRightString(A4[0]-42,22,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=35,bottomMargin=35).build(story,onFirstPage=footer,onLaterPages=footer)
    print('Q4_REPORT_BUILT',len(PdfReader(output).pages),'pages',output)
    return output
if __name__=='__main__':build_report()

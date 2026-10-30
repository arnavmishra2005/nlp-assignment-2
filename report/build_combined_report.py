"""Assemble one combined report from all four supplied question sections.
Requires reportlab, pypdf, uharfbuzz and a Devanagari TrueType font for Q1 text.
Run from root: python report/build_combined_report.py
Uses saved experiment results and the question-specific report sections.
"""
from pathlib import Path
import argparse, csv, html, json, re, hashlib, io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader, PdfWriter
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'report'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def esc(text):return html.escape(str(text))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyGroup',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=7))
styles.add(ParagraphStyle(name='SmallGroup',fontName='Helvetica',fontSize=8,leading=10.5,spaceAfter=5))
styles.add(ParagraphStyle(name='HeadingGroup',fontName='Helvetica-Bold',fontSize=13,leading=16,spaceAfter=9))
styles.add(ParagraphStyle(name='CoverGroup',fontName='Helvetica-Bold',fontSize=22,leading=28,spaceAfter=16))
def p(story,text,small=False):story.append(Paragraph(text,styles['SmallGroup' if small else 'BodyGroup']))
def title(story,text):story.append(Paragraph(text,styles['HeadingGroup']))
def table(story,data,widths):
    t=Table([[Paragraph(esc(x),styles['SmallGroup']) for x in row] for row in data],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.5,colors.grey),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,8)])
def image(story,path,width=490,height=190):
    im=Image(str(path));scale=min(width/im.imageWidth,height/im.imageHeight);im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale;story.extend([im,Spacer(1,5)])
def document(story,label):
    stream=io.BytesIO()
    def footer(c,doc):
        c.setFont('Helvetica',8);c.drawString(42,21,'DSL 504 | Assignment 2 | '+label)
    SimpleDocTemplate(stream,pagesize=A4,leftMargin=42,rightMargin=42,topMargin=35,bottomMargin=35).build(story,onFirstPage=footer,onLaterPages=footer)
    return stream.getvalue()
def q1_section(font):
    pdfmetrics.registerFont(TTFont('Q1Hindi',str(font),subfontIndex=0,shapable=True))
    styles.add(ParagraphStyle(name='HindiGroup',fontName='Q1Hindi',fontSize=9.2,leading=13,spaceAfter=4,shaping=1))
    q=ROOT/'question_1';story=[]
    title(story,'Question 1: Attention-based English-Hindi translation')
    p(story,'Owner: Rudra Mehul Dudhat. Two LSTM encoder-decoder systems compare fixed-state decoding with manually implemented additive attention. Results are documented by the executed notebook, saved histories, translations and attention figures.')
    title(story,'1. Data preparation and architecture')
    p(story,'The IIT Bombay corpus is normalized and tokenized, then empty/malformed pairs, excessive length/ratio and non-Devanagari targets are filtered. From 200,000 sampled raw training pairs, 50,000 cleaned pairs are used. Training pairs have at most 25 tokens per side; retained development/test pairs have at most 30. The filtered evaluation contains 430 development and 1,723 test pairs. Each language vocabulary contains 10,000 entries with PAD/SOS/EOS/UNK, fitted on training only. Maximum training length is 27 including boundary tokens; Hindi training OOV rate is about 9.8%.')
    table(story,[['Setting','Baseline','Attention'],['Architecture','128d embedding; 256d LSTM encoder/decoder; final state initializes decoder','Same sizes; additive scores over all encoder states; masked softmax context feeds decoder'],['Approx. parameters','5.9 million','6.3 million'],['Training (both)','Adam lr 0.002; batch 128; dropout 0.3; seed 0','8 epochs; teacher forcing; clip 1.0; validation-loss-selected checkpoint'],['Decoding (both)','Greedy; maximum 40 steps','PAD/SOS and UNK suppressed as disclosed']],[100,195,195])
    p(story,'Attention uses v^T tanh(W_s s + W_h h_i), excludes padded source positions and forms a weighted encoder-state context. This exposes token-level weights without relying on a prebuilt translation model. The models share major training conditions, but attention adds parameters. Both were trained concurrently on CPU in the supplied run; epoch times are not isolated speed benchmarks.')
    table(story,[['Measured result','Baseline','Attention'],['Best validation loss','4.760','4.654'],['Teacher-forced validation next-token accuracy','0.289','0.302'],['Autoregressive test corpus BLEU (0-100)','0.60','0.92'],['Test word-overlap F1','0.187','0.205']],[280,105,105])
    image(story,q/'results/curves.png',height=180)
    p(story,'Figure Q1.1. Supplied training/validation losses and token accuracies. Teacher-forced accuracy supplies gold preceding tokens; it is not generated-translation accuracy. Both models remain weak under autoregressive decoding.',small=True)
    story.append(PageBreak());title(story,'2. Length, errors and what attention explains')
    table(story,[['Source tokens','Test count','Baseline BLEU','Attention BLEU'],['1-5','10','1.97','2.50'],['6-10','256','1.25','1.14'],['11-15','455','0.56','1.09'],['16+','1,002','0.49','0.82']],[120,90,140,140])
    image(story,q/'results/length.png',height=210)
    p(story,'Figure Q1.2. Supplied length analysis. The accuracy subplot is teacher-forced next-token accuracy; BLEU uses generated translations. The shortest bucket has only ten sentences, so its apparent gain is uncertain. Attention improves BLEU in the 11-15 and 16+ buckets, while the 6-10 bucket is slightly worse. Word-overlap gains do not increase monotonically with length.')
    p(story,'Attention improves word overlap on 926 test sentences, worsens it on 718 and ties on 79. Mean paired gain is 0.0178 with reported sentence-level standard error 0.0024; this is one seed per architecture and does not quantify training-seed uncertainty. Heuristic missing-word flags decrease from 43.0% to 38.2%, while repetition increases from 37.1% to 46.3% and extra-word flags from 29.5% to 34.3%. These rule-based categories are diagnostic indicators, not validated linguistic annotations.')
    p(story,'The historical 30k-pair run used six epochs, vocabulary 8k and batch 64; the current run uses 50k/eight epochs/vocabulary 10k/batch 128. Its higher scores cannot be causally attributed to more data alone. The historical curves/translations are retained as supporting evidence rather than a controlled data-size ablation.')
    p(story,'The supplied attention analysis reports a mean peak weight of 0.32, 86% of steps with no source position above 0.5 and sharpness/word-overlap correlation 0.20. Looking at a relevant source word can still yield an incorrect or incomplete translation. The first/last-third recall statistic concerns Hindi reference position; because word order differs across languages, it does not establish loss of the English source prefix. Likewise, a >=16-token input alone does not diagnose a long-distance dependency error. The six cases below describe observed omissions, substitutions and repetition without making that automatic causal claim.')
    p(story,'Limitations: restricted noisy corpus, word-level vocabularies, small development set, only one model seed, greedy decoding and UNK suppression; the latter hides unknown-token outputs rather than recovering missing information. Neither BLEU below 1 nor visually sharp attention establishes useful translation. Longer training, subwords and other decoding are future options, not verified improvements in this run.')
    p(story,'Sources: Q1 REPORT.md, executed notebook, saved histories, test_translations.csv and figures.',small=True)
    nb=load(q/'NLP_Group_Assignment_2.ipynb');outputs=[]
    for cell in nb['cells']:
        if cell.get('cell_type')=='code' and 'cases = pick_error_cases' in ''.join(cell.get('source',[])):
            outputs.extend(''.join(out.get('text',[])) for out in cell.get('outputs',[]) if 'text' in out)
    text='\n'.join(outputs);cases=[]
    for block in re.split(r'--- case \d+ ---',text)[1:]:
        case={}
        for key in ['English','Reference','Baseline','Attention','Error type']:
            match=re.search(r'^'+re.escape(key)+r'\s*:\s*(.*)$',block,re.M)
            if match:case[key]=match.group(1)
        if case:cases.append(case)
    assert len(cases)==6
    notes=[
        'The unknown name Asaram is replaced; America is omitted. The contrastive connector is partly retained. Sharp focus on an OOV source name does not supply its target translation.',
        'The generated text keeps one day but omits Diwali and the remaining-time relation. The model attends to an unknown event word while producing a generic phrase.',
        'But/she/now are partly aligned, but the key property fine is absent. A plausible alignment is therefore insufficient for a complete translation.',
        'The reporting name is unknown and output repeats fragments. Repetition and an incomplete quotation are directly observed; the heatmap is a clue rather than a causal proof.',
        'The reporting phrase survives, while the Congress/arrogance/media-answer content is lost or replaced. The long source alone is not proof of a specific dependency failure.',
        'The reporting phrase survives but the quote becomes repetitive and semantically unrelated. OOV content and diffuse attention are plausible contributors, not isolated causes.'
    ]
    for i,case in enumerate(cases):
        if i%2==0:story.append(PageBreak());title(story,f'3. Attention error cases {i+1}-{i+2}')
        p(story,'<b>Case '+str(i+1)+'. English:</b> '+esc(case['English']),small=True)
        for key in ['Reference','Baseline','Attention']:
            story.append(Paragraph(esc(key+': '+case[key]),styles['HindiGroup']))
        image(story,q/f'results/error_case_{i+1}.png',height=135)
        p(story,esc(notes[i]),small=True)
    return document(story,'Q1')
def has_ratings():
    path=ROOT/'question_4/evaluation/Human_All_Ratings_Blank_Template.csv'
    if not path.exists():return 0
    data=rows(path)
    keys=['coherence_1_5','grammar_1_5','relevance_1_5','creativity_surprise_1_5']
    return sum(all(x.get(k,'').strip() for k in keys) for x in data)
def cover(spans,q1pages):
    story=[];story.append(Paragraph('NLP Group Assignment 2',styles['CoverGroup']))
    title(story,'Combined technical report - Questions 1 to 4')
    p(story,'DSL 504 - Natural Language Processing<br/>Deadline: 30 October 2026, 11:59 PM IST<br/>Coordinator and final submitter: Arnav Mishra')
    table(story,[['Member','ID','Primary question'],['Rudra Mehul Dudhat','B24DS506','Q1'],['Arnav Mishra','12340330','Q2; integration/submission'],['Vatsal Yadav','B24DS036','Q3'],['Mohak Arya','12341420','Q4, jointly'],['Kanishk Nandeshwar','B24DS010','Q4, jointly']],[190,110,190])
    table(story,[['Section','Combined PDF pages','Content']]+[[k,f'{v[0]}-{v[1]}',{'Question 1':'Models, curves, length analysis and six attention/error cases','Question 2':'Six-page section; completed training/evaluation; seven comparative examples','Question 3':'Revised experiment, generation attention, qualitative analysis and limitations','Question 4':'Models, decoding, generated stories, quantitative results and limitations'}[k]] for k,v in spans.items()],[90,100,300])
    p(story,'The Q2 section is exactly six pages, including its figures and tables. There is no stated six-page cap for the whole group report. Question-specific section reports remain alongside the source; this PDF presents the complete four-question technical narrative. The question-level documents contain their experiment-specific references and limitations.',small=True)
    p(story,'The contribution split and question responsibilities are listed in CONTRIBUTIONS.md.',small=True)
    return document(story,'group overview')
def integration_appendix():
    story=[];title(story,'Experimental summary and limitations')
    table(story,[['Question','Experiment and result','Limitations'],['Q1: translation','50,000 training pairs; LSTM baseline versus additive attention; test BLEU 0.60 versus 0.92','One seed per model; word-level vocabulary; filtered test subset; low translation quality'],['Q2: review ratings','200,000 training reviews; two seeds; test Macro F1 0.6351 for BiGRU versus 0.6706 for RoBERTa','Two seeds; sequence truncation; mixed training topology; pretrained versus non-Transformer comparison'],['Q3: summarization','2,048 training examples; 11,490 test articles; revised model ROUGE-1 1.7506 versus Lead-3 39.9077','Small ordered subset; pervasive repetition; reused test benchmark; no competitive quality claim'],['Q4: story generation','20,000 training stories; RNN/LSTM test perplexity 16.88/13.40; seven challenge prompts','Prompt drift and repetition; pooled metrics include prompts; AI-simulated proxy results are labelled separately; human sheets remain blank as stated by the authors']],[75,235,180])
    p(story,'Q4 length measurements use separate sample sets. The original pooled metrics summarize 35 general-prompt outputs per model and include prompt words. The latest report separately measures all 84 challenge records after removing the exact prompt: 42 records per model, with mean continuation lengths 62.9 words (RNN) and 68.4 (LSTM); two and one continuations respectively are shorter than 20 words. These are descriptive length flags, not proof of incomplete stories. The authors note that an 180-word threshold is unreachable under their 80-new-token cap.')
    p(story,'Across the experiments, attention and pretraining improve some measured tasks but do not guarantee fluent generation, factual accuracy or narrative coherence. Next-token loss and overlap/diversity metrics address different aspects of performance. Qualitative examples and explicit error analysis are needed to interpret the numerical results.')
    p(story,'The supplied source, configurations, executed notebooks, predictions, figures and model-specific report sections document the experiment settings. Reproduction instructions are provided in the main README and each question directory. Dataset/model revisions and package versions are recorded where available; numerical equivalence across environments is not guaranteed.')
    return document(story,'experimental summary')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/Nirmala.ttc'));args=parser.parse_args()
    if not args.font.exists():raise FileNotFoundError('Pass --font with a Devanagari TrueType font (Nirmala/Noto Sans Devanagari). The final PDF embeds it.')
    import runpy
    runpy.run_path(str(ROOT/'question_4/build_report.py'))['build_report']()
    q1=q1_section(args.font)
    parts=[('Question 1',q1),('Question 2',(ROOT/'question_2/report/Q2_Technical_Report.pdf').read_bytes()),('Question 3',(ROOT/'question_3/Final_Report.pdf').read_bytes()),('Question 4',(ROOT/'question_4/Plot_Twist_Generator_Final_Report.pdf').read_bytes())]
    counts={name:len(PdfReader(io.BytesIO(data)).pages) for name,data in parts}
    if counts['Question 2']!=6:raise ValueError('Q2 report exceeds or differs from the checked six-page layout')
    # Cover length is measured before calculating the final section page spans.
    dummy={name:(1,count) for name,count in counts.items()};front=cover(dummy,counts['Question 1']);frontpages=len(PdfReader(io.BytesIO(front)).pages)
    spans={};pos=frontpages+1
    for name,data in parts:spans[name]=(pos,pos+counts[name]-1);pos+=counts[name]
    front=cover(spans,counts['Question 1']);assert len(PdfReader(io.BytesIO(front)).pages)==frontpages
    appendix=integration_appendix();writer=PdfWriter();writer.append(io.BytesIO(front))
    for name,data in parts:
        start=len(writer.pages);writer.append(io.BytesIO(data));writer.add_outline_item(name,start)
    writer.append(io.BytesIO(appendix));writer.add_metadata({'/Title':'NLP Group Assignment 2 - Combined Technical Report','/Author':'NLP Assignment 2 group','/Subject':'Questions 1-4: methodology, experiments, results and limitations'})
    output=REPORT/'Combined_Technical_Report.pdf';writer.write(output)
    check=PdfReader(output);assert len(check.pages)==frontpages+sum(counts.values())+len(PdfReader(io.BytesIO(appendix)).pages)
    manifest={'combined_report':output.relative_to(ROOT).as_posix(),'total_pages':len(check.pages),'section_pages':counts,'section_spans':spans,'q2_page_limit_passed':True,'human_rating_rows_filled':has_ratings(),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    (REPORT/'combined_report_check.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()

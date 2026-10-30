"""Build a six-page Q2 report section from inspected saved measurements.
Run: python question_2/report/build_q2_report.py
Requires reportlab and pypdf. Does not train models or access the test dataset.
"""
from pathlib import Path
import csv, json, html
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'question_2/results'
OUT=Path(__file__).with_name('Q2_Technical_Report.pdf')
def rows(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def esc(s):return html.escape(str(s))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyQ2',fontName='Helvetica',fontSize=9,leading=11.6,spaceAfter=6))
styles.add(ParagraphStyle(name='SmallQ2',fontName='Helvetica',fontSize=7.6,leading=9.3,spaceAfter=4))
styles.add(ParagraphStyle(name='HeadingQ2',fontName='Helvetica-Bold',fontSize=13,leading=16,spaceAfter=9))
story=[]
def para(s,small=False):story.append(Paragraph(s,styles['SmallQ2' if small else 'BodyQ2']))
def title(s):story.append(Paragraph(s,styles['HeadingQ2']))
def table(data,widths=None):
    values=[[Paragraph(esc(v),styles['SmallQ2']) for v in row] for row in data]
    t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8edf3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.5,colors.grey),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4)]));story.append(t);story.append(Spacer(1,6))
def image(name,width=490,height=190):
    im=Image(str(DATA/'figures'/name)); scale=min(width/im.imageWidth,height/im.imageHeight);im.drawWidth=im.imageWidth*scale;im.drawHeight=im.imageHeight*scale;story.append(im);story.append(Spacer(1,5))
def page():story.append(PageBreak())
summary={x['model']:x for x in rows('test_summary.csv')}
val={x['model']:x for x in rows('validation_summary.csv')}
runs=rows('run_details.csv'); examples=rows('qualitative_examples.csv')
notes={
'en_0002917':('Mixed sentiment and practical difficulties','The review combines partial progress with flavor dissatisfaction, damaged foils and difficult capsule preparation. The neutral three-star label fits this mixed evidence. RoBERTa may better combine these qualifications; the positive four-star baseline prediction underweights complaints.'),
'en_0003912':('Rating ambiguity and malformed text','The text is strongly critical, but its recorded label is three stars. RoBERTa matches that label while the baseline predicts two. This is label agreement, not proof that three stars is the uniquely correct interpretation. The malformed apostrophes are present in the saved text.'),
'en_0005862':('Contrast and overriding complaint','The positive opening is reversed by persistent odor, unsuccessful airing and a return. RoBERTa predicts two stars while the baseline predicts three; sensitivity to the contrast and return intent is a plausible explanation, not a demonstrated reasoning process.'),
'en_0008881':('Mixed attributes and unmet expectations','Softness is positive, but excessive warmth and the mismatch with expected towel-like material dominate the complaint. RoBERTa predicts two stars; the baseline predicts three. This suggests different weighting of positive attributes and dissatisfaction.'),
'en_0009772':('Negative utility and recommendation','The dictionary is described as heavy, unhelpful and difficult to use, with an explicit negative recommendation. RoBERTa matches two stars while the baseline gives three. This example concerns failure of the intended use rather than sarcasm.'),
'en_0000916':('Positive title versus short negative body','The title calls the mat nice but the body says it slips and is annoying. The baseline matches three stars; RoBERTa predicts two. The latter may overemphasize the complaint. The short mixed review illustrates uncertainty in mapping wording to exact stars.'),
'en_0005599':('Qualification within strong approval','The reviewer repeatedly likes the product and accepts its weight as compatible with their needs. The baseline matches five stars; RoBERTa gives four. Mention of weight or the qualification so far may contribute, but the explanation remains a hypothesis.')}
# Page 1
title('Question 2: Five-class Amazon review rating prediction')
para('BiGRU with pretrained GloVe and additive attention versus fully fine-tuned RoBERTa with masked mean pooling. Experiments use seeds 42 and 123; first-token pooling is the selected ablation. This section reports inspected saved measurements, without additional model selection after final test evaluation.')
title('1. Data, preprocessing and controlled training')
para('The official SetFit English split contains 200,000 training, 5,000 validation and 5,000 test reviews, balanced across stars 1-5. SetFit supplies bodies but omits titles. Titles were recovered from the original-corpus mirror using exact agreement of ID, body, language and stars, then joined as title + newline + body for both models. Null titles (33 train, one validation) remain empty. Vocabulary and preprocessing fit training only; validation chooses checkpoints. Five exact train/validation text overlaps are disclosed while retaining the required official splits. Reserved test access followed a frozen protocol.')
para('The classic pipeline HTML-unescapes, lowercases and removes tags/URLs, then applies regex tokenization. Minimum frequency 2 yields 32,611 vocabulary entries, capped at 60,000, with PAD/UNK and a 200-token limit. GloVe 6B 100-dimensional vectors cover 87.77% of vocabulary; uncovered vectors use fixed seed 2026. RoBERTa uses original text, byte-level BPE and a 256-token limit with dynamic padding to multiples of eight. Dataset/model revisions and hashes are pinned in config.json and manifests.')
table([['Setting','BiGRU + attention','RoBERTa mean / first'],['Architecture','2 BiGRU layers; 128 hidden/direction; additive attention width 128','roberta-base; all encoder weights trainable; masked mean / first token'],['Head / parameters','Dropout 0.3; linear -> 5; 3,768,497','Dropout 0.2; linear -> 5; 124,058,885'],['Optimizer','Adam, learning rate 0.001','AdamW, learning rate 0.00002; decay 0.01; 8% warmup then linear decay'],['Batch / precision','128; float32','Global microbatch 16; accumulation 2; effective 32; recorded bfloat16'],['Selection / cap','Validation Macro F1; patience 2; at most 8 epochs','Validation Macro F1; patience 1; at most 3 epochs']],[80,205,205])
para('Both use cross-entropy, gradient clipping 1.0 and minimum improvement 0.0001. GloVe is frozen for epoch one then unfrozen. Packed GRU sequences and masked attention exclude padding. RoBERTa mean pooling includes all non-padding states, including special tokens. AdamW excludes biases/LayerNorm from decay. All recorded runs used Tesla T4 hardware; optimizer/scheduler state was retained during active recovery.')
table([['Run','Seed','Epochs / best','Training h','Execution']]+[[x['model'],x['seed'],x['epochs_completed']+' / '+x['best_epoch'],f"{float(x['training_hours']):.2f}",x['world_size']+' GPU(s)'] for x in runs],[145,40,85,70,150])
# Page 2
page();title('2. Learning curves and selection')
image('baseline_seed42_curves.png',height=265)
image('roberta_mean_seed42_curves.png',height=265)
para('Figure 1. Training/validation loss and Macro F1 for seed 42 of the two main models. Equivalent figures for seed 123 and both ablation seeds are supplied in results/figures/. Training metrics accumulate with dropout active; validation uses evaluation mode, so the two curves are not identical evaluation protocols.',small=True)
para('Baseline best epochs were 5/4 for seeds 42/123, with early stopping at epochs 7/6. Mean-pooling RoBERTa selected epochs 3/2; both runs completed the three-epoch cap. Checkpoints use validation Macro F1, not minimum loss or test accuracy. Full histories retain the actual losses and scores rather than only the selected epochs.')
# Page 3
page();title('3. Frozen final test evaluation')
para('Each main model/seed produced one final 5,000-review test prediction file. Inspection checked sample IDs, probability ranges/sums, argmax labels and recomputed metrics, including aggregate means and sample SD (ddof=1). Predictions use argmax with labels 0-4 mapped to stars 1-5. First-token models are evaluated on validation for the ablation, not additionally on test.')
def fmt(model,key,percent=False):
 x=summary[model]; k=100 if percent else 1;return f"{float(x[key+'_mean'])*k:.2f} +/- {float(x[key+'_std'])*k:.2f}" if percent else f"{float(x[key+'_mean']):.4f} +/- {float(x[key+'_std']):.4f}"
table([['Model','Accuracy (%)','Macro F1','MAE'],['BiGRU',fmt('baseline','accuracy',True),fmt('baseline','macro_f1'),fmt('baseline','mae')],['RoBERTa mean',fmt('roberta_mean','accuracy',True),fmt('roberta_mean','macro_f1'),fmt('roberta_mean','mae')]],[100,130,130,130])
para('RoBERTa improves accuracy by 3.34 percentage points and Macro F1 by 0.0355, while reducing MAE by 0.0605 stars. These are descriptive two-seed comparisons, not confidence intervals or significance tests. Smaller observed seed variability does not establish robustness.')
per=rows('per_class_summary.csv')
table([['True rating','BiGRU F1, mean +/- SD','RoBERTa F1, mean +/- SD']]+[[str(star)]+[f"{float(next(x for x in per if x['model']==model and int(x['stars'])==star)['mean']):.4f} +/- {float(next(x for x in per if x['model']==model and int(x['stars'])==star)['std']):.4f}" for model in ['baseline','roberta_mean']] for star in range(1,6)],[90,200,200])
para('RoBERTa improves F1 for every rating; the largest gain is for four stars (about 0.0460). Both models identify extreme ratings more accurately than two-, three- and four-star reviews. Middle-rating wording is often mixed and neighboring rating labels can be ambiguous.')
a=Image(str(DATA/'figures/baseline_confusion_mean.png'),width=237,height=185);b=Image(str(DATA/'figures/roberta_mean_confusion_mean.png'),width=237,height=185);story.append(Table([[a,b]],colWidths=[245,245]))
para('Figure 2. Row-normalized test confusion matrices averaged over seeds: BiGRU (left), RoBERTa (right). Full seed-specific matrices are supplied.',small=True)
para('Adjacent-rating error rates fall from 30.84% to 29.49% of all reviews; distant errors (distance >=2) fall from 5.37% to 3.38%. Adjacent errors comprise 85.16% versus 89.72% of mistakes: the larger share for RoBERTa reflects fewer distant errors, not a higher overall adjacent-error rate. Extreme 1-versus-5 error rates fall from 0.29% to 0.09%.')
# Page 4
page();title('4. Pooling ablation, reproducibility and limitations')
table([['Validation result','Mean pooling','First-token pooling'],['Macro F1 mean +/- SD',f"{float(val['roberta_mean']['macro_f1_mean']):.6f} +/- {float(val['roberta_mean']['macro_f1_std']):.6f}",f"{float(val['roberta_first']['macro_f1_mean']):.6f} +/- {float(val['roberta_first']['macro_f1_std']):.6f}"],['MAE mean +/- SD',f"{float(val['roberta_mean']['mae_mean']):.4f} +/- {float(val['roberta_mean']['mae_std']):.4f}",f"{float(val['roberta_first']['mae_mean']):.4f} +/- {float(val['roberta_first']['mae_std']):.4f}"]],[170,160,160])
image('pooling_ablation.png',height=185)
para('First-token pooling is slightly better on validation in both seeds: mean-minus-first Macro F1 differences are -0.001869 (seed 42) and -0.002076 (seed 123). Its average MAE is lower by 0.0013. These small differences do not establish a universal pooling advantage. The required mean-pooling architecture remains the main final-test model; the ablation does not trigger test-driven architecture replacement.')
para('Execution topology limits pooling-only attribution. Mean seed 42 used one GPU, while mean seed 123 and both first-token runs used two-GPU DistributedDataParallel. Global microbatch 16, accumulation 2 and effective batch 32 were preserved; gradient/communication probes passed, but rank-specific dropout streams and numerical reduction order can alter trajectories. The seed-123 pair shares topology; the seed-42 pair does not. Both baselines used one GPU. Actual topology, precision and source provenance are retained per run.')
para('Training and evaluation were performed in the recorded Kaggle environment; CPU finalization only inspected and packaged saved files. It did not retrain or access the test corpus. Six training archives retain the selected checkpoints; the compact submission component excludes full data/caches/checkpoints. The final test protocol hash is recorded in FROZEN_PROTOCOL.json, and generated analysis can reuse its saved predictions.')
para('Limitations include only two seeds, a single review domain, rare/OOV tokens, bounded sequence lengths, five known train/validation text overlaps, and ambiguity between neighboring star labels. RoBERTa also benefits from extensive external pretraining, so this comparison does not isolate architecture alone. No calibration, sarcasm understanding, deployment safety or out-of-domain robustness was established. Improvements should be selected on new training/validation experiments, not the final test results.')
para('Conclusion: RoBERTa mean pooling improves the measured rating-classification and ordinal-error metrics over this GloVe-BiGRU baseline. First-token pooling has a small validation advantage in the present ablation. Neither finding establishes a general superiority across datasets, execution environments or untested hyperparameters.')
# Page 5
page();title('5. Baseline-wrong / RoBERTa-right examples')
para('Fixed seed 42, selected by ascending review ID: five of 500 available baseline-wrong/RoBERTa-right cases. Labels and predictions are unedited; interpretation below is a text-supported hypothesis, not a causal account of internal reasoning. Full texts are also retained in qualitative_examples.csv.',small=True)
for x in examples:
 if x['comparison']!='baseline_wrong_roberta_right':continue
 category,note=notes[x['id']]
 para(f"<b>{esc(x['id'])}: true {x['true_stars']}; BiGRU {x['baseline_stars']}; RoBERTa {x['roberta_stars']}</b>")
 para('<b>Title:</b> '+esc(x['title'])+'<br/><b>Review:</b> '+esc(x['body']),small=True)
 para('<b>'+esc(category)+'.</b> '+esc(note),small=True)
# Page 6
page();title('6. Reverse cases and token-level importance')
para('Two of 362 available baseline-right/RoBERTa-wrong cases, again in ascending review-ID order for seed 42. The stronger aggregate model still fails on individual reviews.',small=True)
for x in examples:
 if x['comparison']!='baseline_right_roberta_wrong':continue
 category,note=notes[x['id']]
 para(f"<b>{esc(x['id'])}: true {x['true_stars']}; BiGRU {x['baseline_stars']}; RoBERTa {x['roberta_stars']}</b>")
 para('<b>Title:</b> '+esc(x['title'])+'<br/><b>Review:</b> '+esc(x['body']),small=True)
 para('<b>'+esc(category)+'.</b> '+esc(note),small=True)
image('roberta_token_importance.png',height=180)
para('Figure 3. Absolute gradient x input importance for the predicted-class logit of the fixed RoBERTa model on en_0002917. Scores are normalized across all retained tokens, including special tokens. The displayed subset is not the complete review; full scores are in token_importance.csv.',small=True)
para('In the saved scores, the contrast marker but (position 32) has importance 0.0398; capsule 0.0368, foil 0.0334, the fo subword of foils 0.0322, difficult 0.0264 and try 0.0261 also receive substantial weight. This sensitivity is consistent with practical difficulties and mixed sentiment, but does not establish how the model combined them or prove negation understanding. Tokens for the first occurrence of the negation have smaller scores than some context tokens. The example has 104 retained tokens, below the 256-token limit; truncation is therefore not the explanation for this particular difference. Other long reviews may lose text.')
para('<b>Reproduction/evidence:</b> question_2/README.md and KAGGLE_RUN_GUIDE.md describe exact phases; config.json, manifests, environment records, training_records/, final prediction CSVs and metric JSONs retain run-level provenance. Qualitative interpretations are based on the saved review text and model predictions.',small=True)
para('<b>Sources:</b> Assignment PDF pp. 5-7; SetFit/amazon_reviews_multi_en and buruzaemon original-corpus mirror (exact IDs/revisions in config.json); Stanford GloVe 6B; FacebookAI/roberta-base. Full source links and recorded settings are retained in the component README/configuration.',small=True)
def footer(canvas,doc):
 canvas.setFont('Helvetica',8);canvas.drawString(42,22,'DSL 504 | Assignment 2 | Question 2');canvas.drawRightString(A4[0]-42,22,f'{doc.page} / 6')
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=36)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
count=len(PdfReader(OUT).pages)
if count!=6:raise RuntimeError(f'Expected exactly six pages, got {count}; adjust layout before submission')
# Preserve an annotated copy; keep the inspected raw selection unchanged.
annotated=[]
for item in examples:
 item=dict(item); item['human_error_category'],item['human_analysis']=notes[item['id']]; annotated.append(item)
with OUT.with_name('Q2_qualitative_examples_annotated.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(annotated[0]));writer.writeheader();writer.writerows(annotated)
OUT.with_name('Q2_report_check.json').write_text(json.dumps({'pages':count,'maximum_pages':6,'qualitative_forward_cases':5,'qualitative_reverse_cases':2,'built_from':'inspected Q2_evaluate_results.zip measurements','model_training_performed':False},indent=2),encoding='utf-8')
print('REPORT_PAGE_CHECK_PASSED',OUT,count,'pages')

"""Generate notebooks with exactly three code cells: configure, setup, launch."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, index):
    value = {"cell_type": kind, "metadata": {}, "id": f"cell-{index:03}", "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


CONFIG = r"""# Cell 1: configuration and installed-environment checks (no training).
from pathlib import Path
import importlib.metadata, json, os, platform, shutil, subprocess, sys
PHASE = __PHASE__
SOURCE_PATH = None  # Optional exact q2_source.zip OR extracted question_2 folder.
DATA = None         # prepare: optional raw folder; others: optional prepare phase ZIP/folder.
PREVIOUS = []       # Optional explicit predecessor archives. Auto-discovery requires one per phase.
RESUME_FROM = None  # Optional own same-phase FAILED archive.
GLOVE_PATH = None   # Optional attached official GloVe ZIP/txt for prepare.
INPUTS = Path('/kaggle/input')
OUTPUT = Path('/kaggle/temp/q2-artifacts')
PUBLISH = Path('/kaggle/working')
MIN_FREE_GIB = 8
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TQDM_DISABLE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'
os.environ['PIP_PROGRESS_BAR'] = 'off'
print('PHASE:', PHASE, '\nPython:', sys.version, '\nPlatform:', platform.platform(), flush=True)
for name in ('torch','transformers','datasets','numpy','pandas','scikit-learn'):
    try:
        print(name, importlib.metadata.version(name), flush=True)
    except importlib.metadata.PackageNotFoundError:
        print(name, 'MISSING; non-CUDA dependencies will be checked in cell 2', flush=True)
smi = shutil.which('nvidia-smi')
if smi:
    subprocess.run([smi, '--query-gpu=index,name,memory.total', '--format=csv'], check=True)
else:
    print('nvidia-smi: not found', flush=True)
probe = '\n'.join([
    'import sys, torch',
    'phase=sys.argv[1]',
    "required=phase not in ('prepare','finalize')",
    "print('torch CUDA available:',torch.cuda.is_available(),flush=True)",
    'if required and not torch.cuda.is_available():',
    "    raise RuntimeError('GPU phase: enable a Kaggle NVIDIA GPU before execution')",
    'if not required and torch.cuda.is_available():',
    "    raise RuntimeError('CPU phase: set Accelerator=None to preserve quota')",
    'if required:',
    "    print('GPU:',torch.cuda.get_device_name(0),'visible count:',torch.cuda.device_count(),flush=True)",
    "    x=torch.ones((4,4),device='cuda')",
    '    assert float((x@x).sum().item())==64.0',
    "    print('GPU_OPERATION_PASSED; initial device probe only',flush=True)",
    "if phase.startswith('roberta_'):",
    "    if torch.cuda.device_count()<2: raise RuntimeError('DUAL_GPU_REQUIRED: select Kaggle GPU T4 x2')",
    "    for index in range(2):",
    "        x=torch.ones((4,4),device=f'cuda:{index}')",
    "        assert float((x@x).sum().item())==64.0",
    "        print('GPU_OPERATION_PASSED',index,torch.cuda.get_device_name(index),flush=True)",
    "    print('DUAL_GPU_REQUIRED_AND_AVAILABLE; 2 workers, local batch8, accumulation2, effective batch32',flush=True)",
])
subprocess.run([sys.executable,'-u','-c',probe,PHASE],check=True)
free=shutil.disk_usage('/kaggle/working').free/2**30
print('Free working disk GiB:',round(free,2),flush=True)
if free < MIN_FREE_GIB:
    raise RuntimeError('Insufficient free disk for artifacts plus the phase archive')
"""

SETUP = r"""# Cell 2: locate/copy source, install only needed packages, verify prerequisites and plan.
import zipfile
CODE = Path('/kaggle/temp/q2-code')
CODE.mkdir(parents=True,exist_ok=True)
if SOURCE_PATH:
    source = Path(SOURCE_PATH)
else:
    archives = sorted(INPUTS.rglob('q2_source.zip'))
    if len(archives)==1:
        source=archives[0]
    elif not archives:
        roots=sorted(set(p.parent for p in INPUTS.rglob('config.json')
                         if 'q2-code' not in p.parts and (p.parent/'kaggle_start.py').exists()))
        if len(roots)!=1:
            raise RuntimeError(f'Ambiguous/missing source inputs: {roots}. Set SOURCE_PATH explicitly.')
        source=roots[0]
    else:
        raise RuntimeError('Multiple source ZIPs. Set SOURCE_PATH to one exact version.')
PROJECT=CODE/'question_2'
if source.is_file():
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            if not (CODE/name).resolve().is_relative_to(CODE.resolve()):
                raise ValueError('Unsafe source archive entry')
        archive.extractall(CODE)
else:
    shutil.copytree(source,PROJECT,dirs_exist_ok=True)
if not (PROJECT/'kaggle_start.py').exists():
    raise RuntimeError('Source layout invalid: expected question_2/kaggle_start.py')
required_files = ['deliver_phase.py', 'compact_results.py', 'q2/storage.py', 'q2/compatibility.py',
                  'q2/distributed.py', 'q2/train_distributed.py']
missing = [name for name in required_files if not (PROJECT/name).is_file()]
if missing:
    raise RuntimeError('OUTDATED_SOURCE_DATASET: missing '+', '.join(missing)+
                       '. Update the Kaggle source dataset with the current root-level q2_source.zip, '
                       'attach its latest version, and start a fresh session. Training has not started.')
package_init = (PROJECT/'q2/__init__.py').read_text(encoding='utf-8')
if 'TQDM_DISABLE' not in package_init:
    raise RuntimeError('OUTDATED_SOURCE_DATASET: update q2_source.zip to the current plain-text logging version before training.')
print('SOURCE_VERSION_CHECK_PASSED; plain-text progress and dual-GPU RoBERTa support enabled', flush=True)
if str(PROJECT) not in sys.path:
    sys.path.insert(0,str(PROJECT))
# Standard-library-only helper; safe before dependency installation.
from q2.process import run_streaming
run_streaming([sys.executable,'-u',PROJECT/'setup_environment.py','--install-missing'],cwd=PROJECT)
COMMAND=[sys.executable,'-u',str(PROJECT/'kaggle_start.py'),'--phase',PHASE,
         '--config',str(PROJECT/'config.json'),'--inputs',str(INPUTS),'--output',str(OUTPUT),
         '--min-free-gib',str(MIN_FREE_GIB)]
if DATA: COMMAND+=['--data',str(DATA)]
for previous in PREVIOUS: COMMAND+=['--previous',str(previous)]
if RESUME_FROM: COMMAND+=['--resume-from',str(RESUME_FROM)]
if GLOVE_PATH: COMMAND+=['--glove-path',str(GLOVE_PATH)]
def run_phase(command, log_path=None):
    try:
        return run_streaming(command,log_path=log_path,cwd=PROJECT)
    except Exception:
        failed=OUTPUT/PHASE/f'Q2_{PHASE}_FAILED.zip'
        if failed.exists():
            run_streaming([sys.executable,'-u',PROJECT/'deliver_phase.py','--archive',failed,'--destination',PUBLISH],cwd=PROJECT)
        raise
run_phase(COMMAND+['--setup-only'])
print('SETUP_CHECK_PASSED\nPHASE:',PHASE,'\nPREREQUISITES_VERIFIED',flush=True)
"""

LAUNCH = r"""# Cell 3: launch ONE phase, stream logs, inspect outputs, then report success.
# Use Save Version -> Save & Run All from a fresh session. Do not launch another
# complete saved run after starting this cell interactively.
run_phase(COMMAND,log_path=OUTPUT/PHASE/'notebook_launcher.log')
archive=OUTPUT/PHASE/f'Q2_{PHASE}_results.zip'
if not archive.exists():
    raise RuntimeError('Expected phase archive missing; execution is not complete')
run_streaming([sys.executable,'-u',PROJECT/'inspect_results.py',archive,'--expected-phase',PHASE,
               '--config',PROJECT/'config.json'],cwd=PROJECT)
execution=json.loads((OUTPUT/PHASE/'execution.json').read_text())
if execution['status']!='COMPLETE':
    raise RuntimeError('Execution status is not COMPLETE')
run_streaming([sys.executable,'-u',PROJECT/'deliver_phase.py','--archive',archive,'--destination',PUBLISH],cwd=PROJECT)
archive=PUBLISH/f'Q2_{PHASE}_results.zip'
print('PHASE_COMPLETE:',PHASE,'\nDOWNLOAD:',archive,'\nINSPECTION_PASSED',flush=True)
print('Return this archive for inspection before proceeding to the next phase.',flush=True)
"""


def main():
    phases = [('00_kaggle_workflow.ipynb','prepare'),('01_prepare_data_CPU.ipynb','prepare'),
              ('01b_smoke_GPU.ipynb','smoke')]
    index=2
    for model in ('baseline','roberta_mean','roberta_first'):
        for seed in (42,123):
            phase=f'{model}_seed{seed}'
            phases.append((f'{index:02}_{phase}.ipynb',phase));index+=1
    phases += [('08_final_evaluation_and_report.ipynb','evaluate'),('09_finalize_CPU.ipynb','finalize')]
    for filename,phase in phases:
        header=f'# Q2 staged Kaggle workflow: {phase}\n\nExactly three code cells: configure/check, locate/setup, launch/inspect. See KAGGLE_RUN_GUIDE.md for required inputs. Run one phase, return its archive, inspect, then continue. No test access before evaluate. Smoke outputs are separate correctness checks, never assignment results.\n\nFull plan: 6 training runs; seeds 42 and 123; BiGRU up to 8 epochs each, mean-pooling and first-token RoBERTa up to 3 each (28 maximum epochs total). Early stopping may shorten runs. Completion does not imply a performance target was achieved.'
        if phase.startswith('roberta_'):
            header += '\n\nRequires GPU T4 x2. Two DDP workers use 8 reviews per GPU and 2 accumulation steps: unchanged effective batch32. Same original global shuffle, schedule and full official data. Both GPUs and gradient scaling are checked before full training. Use the updated source ZIP.'
        cells=[cell('markdown',header,0),cell('code',CONFIG.replace('__PHASE__',repr(phase)),1),
               cell('code',SETUP,2),cell('code',LAUNCH,3)]
        notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                  'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
        target=ROOT/'notebooks'/filename;target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(notebook,indent=1,ensure_ascii=False),encoding='utf-8')
    print('Generated',len(phases),'notebooks, each with exactly 3 code cells')


if __name__=='__main__':
    main()

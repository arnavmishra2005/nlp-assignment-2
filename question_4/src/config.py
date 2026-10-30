from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

DATASET_NAME = "roneneldan/TinyStories"
NUM_TRAIN = 20000
NUM_VAL = 1000
NUM_TEST = 1000

MIN_FREQ = 5
MAX_VOCAB = 10000
MAX_LEN = 200

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

BATCH_SIZE = 64

CHECKPOINT_DIR = ROOT / "checkpoints"
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.2
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
MAX_EPOCHS = 10
PATIENCE = 3

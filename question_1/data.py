import os, re, pickle, random
from collections import Counter
import numpy as np

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_pairs(n_train_raw=100000, seed=0):
    from datasets import load_dataset
    raw = load_dataset("cfilt/iitb-english-hindi")
    pairs = lambda split: [(x["translation"]["en"], x["translation"]["hi"]) for x in split]
    train = raw["train"].shuffle(seed=seed).select(range(n_train_raw))
    return pairs(train), pairs(raw["validation"]), pairs(raw["test"])


def normalize(text):
    text = text.replace("‌", "").replace("‍", "")
    return " ".join(text.split())

def tokenize_en(text):
    return re.findall(r"\w+|[^\w\s]", normalize(text).lower())

def tokenize_hi(text):
    # \w would cut hindi words at the vowel signs, so split on spaces and punctuation instead
    return re.findall(r"[^\s।,.!?;:\"'()\-–]+|[।,.!?;:\"'()\-–]", normalize(text))


def is_good_pair(en, hi, max_len):
    if not en or not hi or len(en) > max_len or len(hi) > max_len:
        return False
    ratio = len(en) / len(hi)
    has_hindi = any("ऀ" <= ch <= "ॿ" for w in hi for ch in w)
    return 1 / 3 <= ratio <= 3 and has_hindi

def clean_pairs(pairs, max_len):
    tokenized = [(tokenize_en(e), tokenize_hi(h)) for e, h in pairs]
    return [(e, h) for e, h in tokenized if is_good_pair(e, h, max_len)]


def build_vocab(token_lists, max_size):
    counts = Counter(w for tokens in token_lists for w in tokens)
    words = SPECIAL + [w for w, _ in counts.most_common(max_size - len(SPECIAL))]
    return {w: i for i, w in enumerate(words)}, words

def encode(tokens, stoi):
    return [SOS] + [stoi.get(w, UNK) for w in tokens] + [EOS]

def pad(sequences):
    lengths = np.array([len(s) for s in sequences])
    padded = np.zeros((len(sequences), lengths.max()), dtype=np.int64)
    for i, s in enumerate(sequences):
        padded[i, :len(s)] = s
    return padded, lengths


def prepare(n_train=50000, n_raw=200000, max_len=25, vocab_size=10000, seed=0, verbose=True):
    cache = os.path.join(RESULTS, f"data_{n_train}.pkl")
    if os.path.exists(cache):
        data = pickle.load(open(cache, "rb"))
    else:
        train_raw, dev_raw, test_raw = load_pairs(n_raw, seed=seed)
        train = clean_pairs(train_raw, max_len)
        random.Random(seed).shuffle(train)
        train = train[:n_train]
        dev, test = clean_pairs(dev_raw, 30), clean_pairs(test_raw, 30)
        en_stoi, en_itos = build_vocab([e for e, _ in train], vocab_size)
        hi_stoi, hi_itos = build_vocab([h for _, h in train], vocab_size)
        data = dict(train=train, dev=dev, test=test, en_stoi=en_stoi, en_itos=en_itos,
                    hi_stoi=hi_stoi, hi_itos=hi_itos, n_train_raw=len(train_raw))
        os.makedirs(RESULTS, exist_ok=True)
        pickle.dump(data, open(cache, "wb"))
    if verbose:
        print_report(data)
    return data


def print_report(data):
    train = data["train"]
    unk_rate = lambda side, stoi: sum(w not in stoi for p in train for w in p[side]) / sum(len(p[side]) for p in train)
    print("training pairs used     :", len(train))
    print("dev / test pairs        :", len(data["dev"]), "/", len(data["test"]))
    print("English vocab size      :", len(data["en_itos"]))
    print("Hindi vocab size        :", len(data["hi_itos"]))
    print("max English length      :", max(len(e) for e, _ in train) + 2, "(with <SOS> <EOS>)")
    print("max Hindi length        :", max(len(h) for _, h in train) + 2, "(with <SOS> <EOS>)")
    print(f"unknown word rate       : English {unk_rate(0, data['en_stoi']):.1%}, Hindi {unk_rate(1, data['hi_stoi']):.1%}")


if __name__ == "__main__":
    data = prepare()
    en, hi = data["train"][1]
    print("\nEnglish:", ["<SOS>"] + en + ["<EOS>"])
    print("Hindi  :", ["<SOS>"] + hi + ["<EOS>"])
    print("ids    :", encode(en, data["en_stoi"]), encode(hi, data["hi_stoi"]))
    print(pad([encode(e, data["en_stoi"]) for e, _ in data["train"][:3]]))

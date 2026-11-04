"""HotpotQA EM/F1 and GSM8K numeric accuracy, as in the original experiments."""
import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation


def normalize_answer(text):
    text = str(text).lower().strip()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return int(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction, reference):
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    special = {"yes", "no", "noanswer"}
    if (set(p) & special) != (set(r) & special):
        return 0.0
    if not p or not r:
        return float(p == r)
    overlap = sum((Counter(p) & Counter(r)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def extract_number(value):
    s = str(value).strip()
    s = re.sub(r"^(?:final\s+)?answer\s*:\s*", "", s, flags=re.I)
    s = re.sub(r"^####\s*", "", s)
    s = s.replace(",", "").replace("$", "").strip().rstrip(".").strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", s):
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def numeric_accuracy(prediction, reference):
    p, r = extract_number(prediction), extract_number(reference)
    return int(p is not None and r is not None and p == r)


def score(benchmark, prediction, reference):
    if benchmark == "gsm8k":
        a = numeric_accuracy(prediction, reference)
        return {"accuracy": a, "em": a, "f1": None}
    return {"accuracy": exact_match(prediction, reference),
            "em": exact_match(prediction, reference),
            "f1": token_f1(prediction, reference)}

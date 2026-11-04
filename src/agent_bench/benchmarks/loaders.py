"""Dataset loading: identical HotpotQA ordering and GSM8K seeded shuffle to original scripts."""
import re
from datasets import load_dataset


def load_examples(benchmark: str, n: int, seed: int = 42):
    if n < 1:
        raise ValueError("--num-examples must be positive")
    if benchmark == "hotpotqa":
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"validation[:{n}]")
        return [{
            "id": r["id"], "question": r["question"], "answer": r["answer"],
            "documents": [{"title": title, "content": " ".join(sentences)}
                          for title, sentences in zip(r["context"]["title"], r["context"]["sentences"])],
            "type": r["type"], "level": r["level"],
        } for r in ds]
    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        rows = []
        for i, r in enumerate(ds):
            match = re.search(r"####\s*([^\n]+)", r["answer"])
            if not match:
                raise ValueError(f"Missing GSM8K gold answer at index {i}")
            rows.append({"id": f"gsm8k-{seed}-{i}", "question": r["question"],
                         "answer": match.group(1).strip(), "documents": [],
                         "type": "math_word_problem", "level": "gsm8k"})
        return rows
    raise ValueError(f"Unknown benchmark: {benchmark}")

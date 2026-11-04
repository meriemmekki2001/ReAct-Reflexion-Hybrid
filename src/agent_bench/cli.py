"""Command-line evaluation; checkpoints results after each agent invocation."""
import argparse
import os
import time
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from agent_bench.benchmarks.loaders import load_examples
from agent_bench.benchmarks.metrics import score
from agent_bench.runner import AGENTS, run_agent


def evaluate(benchmark, n=10, seed=42, agents=None, results_dir=None, start_index=0):
    agents = agents or list(AGENTS)
    output = Path(results_dir or f"results/{benchmark}")
    output.mkdir(parents=True, exist_ok=True)
    examples = load_examples(benchmark, n, seed)
    rows = []
    for example in tqdm(examples[start_index:], desc=f"{benchmark} questions"):
        for name in agents:
            started = time.perf_counter()
            try:
                result = run_agent(name, example, benchmark)
                current = score(benchmark, result["answer"], example["answer"])
                first = score(benchmark, result["first_answer"], example["answer"])
                row = dict(id=example["id"], agent=name, question=example["question"],
                           reference=example["answer"], prediction=result["answer"],
                           first_answer=result["first_answer"],
                           accuracy=current["accuracy"], first_accuracy=first["accuracy"],
                           em=current["em"], f1=current["f1"],
                           first_em=first["em"], first_f1=first["f1"],
                           recovered=int(first["accuracy"] == 0 and current["accuracy"] == 1),
                           regressed=int(first["accuracy"] == 1 and current["accuracy"] == 0),
                           tool_calls=result["tool_calls"], trials=result["trials"],
                           reflection_triggered=result["reflection_triggered"],
                           status="success", error="")
            except Exception as exc:
                print(f"\nError [{name}] on {example['id']}: {exc}")
                traceback.print_exc()
                row = dict(id=example["id"], agent=name, question=example["question"],
                           reference=example["answer"], prediction="", first_answer="",
                           accuracy=None, first_accuracy=None, em=None, f1=None,
                           first_em=None, first_f1=None, recovered=None, regressed=None,
                           tool_calls=None, trials=None, reflection_triggered=None,
                           status="error", error=str(exc))
            row["time_seconds"] = time.perf_counter() - started
            row["question_type"] = example["type"]
            row["level"] = example["level"]
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    df = pd.DataFrame(rows)
    if df.empty:
        print("No examples selected")
        return
    good = df[df.status == "success"].copy()
    summary_rows = []
    for name in agents:
        all_rows = df[df.agent == name]
        ok = good[good.agent == name]
        failures = len(all_rows) - len(ok)
        data = dict(agent=name, Attempted=len(all_rows), Evaluated=len(ok), Errors=failures)
        if len(ok):
            data.update(Accuracy=ok.accuracy.mean(), First_Accuracy=ok.first_accuracy.mean(),
                        EM=ok.em.mean(), First_EM=ok.first_em.mean(),
                        F1=ok.f1.mean(), First_F1=ok.first_f1.mean(),
                        Avg_Time=ok.time_seconds.mean(), Avg_Tool_Calls=ok.tool_calls.mean(),
                        Avg_Trials=ok.trials.mean(), Reflection_Rate=ok.reflection_triggered.mean(),
                        Recovered=int(ok.recovered.sum()), Regressed=int(ok.regressed.sum()))
            failed_first = int((ok.first_accuracy == 0).sum())
            data["Recovery_Rate"] = data["Recovered"] / failed_first if failed_first else float("nan")
        summary_rows.append(data)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False)
    print(f"\n========== {benchmark.upper()} SUMMARY ==========\n")
    print(summary.round(3).to_string(index=False))
    print(f"\nResults saved to: {output}")


def main():
    parser = argparse.ArgumentParser(description="ReAct/Reflexion benchmark evaluation")
    parser.add_argument("--benchmark", choices=["hotpotqa", "gsm8k"], required=True)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="GSM8K shuffle seed; HotpotQA uses first N")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    evaluate(args.benchmark, args.num_examples, args.seed, args.agents, args.results_dir, args.start_index)

if __name__ == "__main__":
    main()

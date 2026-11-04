# ReAct–Reflexion: Reasoning Benchmark Experiments

A compact, reproducible **LangChain** research project comparing four LLM configurations on two reasoning benchmarks. Prepared for the master's thesis *Enhancing LLM Reasoning through Agent Frameworks*.

**Agents:** Standalone Baseline · ReAct · Reflexion · Hybrid (ReAct + Reflexion)

**Benchmarks:** HotpotQA (multi-hop factual QA) · GSM8K (grade-school math word problems)

> This repository is a cleaned-up refactor of an experimental implementation, not an official implementation of the ReAct or Reflexion papers. It preserves the original agent control flow, prompts, and evaluation conventions. No performance numbers are guaranteed.

## Quick start

**Python 3.11 recommended.** An OpenAI API key and internet access to download datasets are required. API calls incur costs.

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Copy `.env.example` to `.env` and enter `OPENAI_API_KEY`. The default model is `gpt-4.1-mini`; override it using `OPENAI_MODEL` in `.env`. **Never commit `.env` or share your API key.**

### Smoke test (2 questions)

```bash
python -m agent_bench.cli --benchmark hotpotqa --num-examples 2
python -m agent_bench.cli --benchmark gsm8k --num-examples 2 --seed 42
```

### Full sample (100 questions, four agents)

```bash
python -m agent_bench.cli --benchmark hotpotqa --num-examples 100
python -m agent_bench.cli --benchmark gsm8k --num-examples 100 --seed 42
```

### Run only one or two agents

```bash
python -m agent_bench.cli --benchmark gsm8k --num-examples 10 --agents ReAct Hybrid
python -m agent_bench.cli --benchmark hotpotqa --num-examples 10 --agents Baseline Reflexion
```

Results are saved to `results/hotpotqa/` or `results/gsm8k/` as `comparison.csv` (per-question outputs) and `summary.csv` (aggregate metrics). **A run overwrites CSVs in its selected results directory.** Use `--results-dir` for independent runs.

## Methods

| Configuration | Tools | Reflection | Attempts |
| --- | --- | --- | --- |
| Baseline | No | No | 1 |
| ReAct | Yes | No | 1 ReAct trajectory, up to 6 tool-calling steps |
| Reflexion | No | Yes, unconditional | 2 LLM answers |
| Hybrid | Yes | Yes, unconditional | 2 ReAct trajectories |

The *same underlying model* is used by all four agents. **Baseline and Reflexion do not receive HotpotQA context documents**; ReAct and Hybrid retrieve from candidate documents through tools. This is an intentional carry-over from the original experiment and **not a controlled comparison with equal evidence access**. The previous Context Baseline is intentionally excluded at the request of the project owner. The Hybrid uses reflection after every first attempt, even if the first answer was correct.

### Task-specific tools

- **HotpotQA:** `search_documents` (BM25) and `read_document` (exact title), over the question's candidate context paragraphs. These are *not* external Wikipedia searches.
- **GSM8K:** `calculator` (restricted arithmetic expression evaluator). The model receives the problem text, not the dataset's worked solution or gold answer.

The original ReAct tool loop is shared across both benchmarks. A GSM8K-specific prompt and calculator replace document retrieval; no BM25 index is initialized for empty document lists.

## Metrics

- **HotpotQA:** normalized exact match (EM) and token-level F1.
- **GSM8K:** exact numeric accuracy after stripping answer prefixes, currency signs, and thousands separators. This is a strict final-number check, not a judge model.
- **Both:** first-attempt accuracy, recovered and regressed questions, recovery rate, mean runtime, tool calls, trials, reflection rate, attempted/success/error counts.

A recovery is an initially incorrect prediction that becomes correct on the second attempt. A regression is an initially correct prediction that becomes incorrect. Execution errors are retained as error rows and **excluded from accuracy averages**; always inspect `Errors` before interpreting a result. Time includes model calls and network latency, so it is not a hardware-independent compute metric.

## Repository layout

```text
src/agent_bench/
  agents/              # Baseline, ReAct, Reflexion, Hybrid + answer format
  tools/               # BM25 retrieval and arithmetic calculator
  benchmarks/          # Dataset loaders, prompts, evaluation metrics
  llm.py               # Shared ChatOpenAI configuration
  runner.py            # Dispatches four agents on either benchmark
  cli.py               # Experiment CLI and CSV summaries
tests/                 # Offline smoke/unit tests
docs/                  # Methodology and reproducibility notes
```

## Reproducibility and limitations

- **HotpotQA:** `hotpotqa/hotpot_qa`, `distractor`, first N validation questions (same selection as original HotpotQA code). The `--seed` option does not shuffle HotpotQA.
- **GSM8K:** `openai/gsm8k`, `main`, test split shuffled with `--seed` (42 by default), first N questions (same selection as original GSM8K code).
- Model temperature is 0, but API outputs are not guaranteed to be fully deterministic. Record the actual model ID, library versions, sample size, seed, run date, and API costs in your thesis.
- This refactor retains the original agent logic but is **not guaranteed to reproduce prior numerical results exactly**: package/model updates, prompt changes in any dependency, or nondeterministic outputs can affect scores.
- A fairer evidence-access comparison would also include a context-aware standalone baseline; this four-agent version deliberately omits it.
- Reflexion here is **self-critique without oracle feedback**. It does not receive correctness labels or gold answers. This differs from some Reflexion research setups.
- The GSM8K calculator offers an extra task-specific capability to tool-using agents; cross-benchmark results should be interpreted as task-adapted agent comparisons, not as a pure architecture-only effect.

For details, see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## References

- Yao et al. (2023), [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).
- Shinn et al. (2023), [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366).
- Yang et al. (2018), [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600).
- Cobbe et al. (2021), [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168).

## License

MIT (see `LICENSE`). Review third-party dataset/model terms separately.

"""Four agent configurations, unchanged core algorithms, benchmark-specific tools/prompts."""
from agent_bench.agents.baseline import baseline_answer
from agent_bench.agents.react_agent import react_answer
from agent_bench.agents.reflexion_agent import reflexion_answer
from agent_bench.agents.hybrid_agent import hybrid_answer
from agent_bench.tools.calculator import create_gsm8k_tools
from agent_bench.benchmarks.gsm8k_prompts import (
    MATH_ANSWER, MATH_REACT, MATH_REFLECTION, MATH_HYBRID_REFLECTION
)

AGENTS = ("Baseline", "ReAct", "Reflexion", "Hybrid")


def run_agent(name, example, benchmark):
    question, documents = example["question"], example["documents"]
    math = benchmark == "gsm8k"
    tool_factory = create_gsm8k_tools if math else None
    react_prompt = MATH_REACT if math else None
    if name == "Baseline":
        answer = baseline_answer(question, system_prompt_override=MATH_ANSWER if math else None)
        return {"answer": answer, "first_answer": answer, "trials": 1,
                "tool_calls": 0, "reflection_triggered": False}
    if name == "ReAct":
        answer, trajectory = react_answer(
            question, documents, return_trajectory=True,
            tool_factory=tool_factory, system_prompt_override=react_prompt)
        return {"answer": answer, "first_answer": answer, "trials": 1,
                "tool_calls": len(trajectory), "reflection_triggered": False}
    if name == "Reflexion":
        result = reflexion_answer(
            question, return_details=True,
            answer_prompt_override=MATH_ANSWER if math else None,
            reflection_prompt_override=MATH_REFLECTION if math else None)
    elif name == "Hybrid":
        result = hybrid_answer(
            question, documents, return_details=True,
            tool_factory=tool_factory, react_prompt_override=react_prompt,
            reflection_prompt_override=MATH_HYBRID_REFLECTION if math else None)
    else:
        raise ValueError(f"Unknown agent: {name}")
    return {"answer": result["answer"], "first_answer": result["history"][0]["answer"],
            "trials": result["trials"], "tool_calls": result.get("tool_calls", 0),
            "reflection_triggered": True}

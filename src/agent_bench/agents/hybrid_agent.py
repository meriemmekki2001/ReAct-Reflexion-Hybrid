
from langchain_core.messages import SystemMessage, HumanMessage

from agent_bench.llm import get_llm
from agent_bench.agents.react_agent import react_answer


# ==========================================
# REFLECTION PROMPT
# ==========================================

HYBRID_REFLECTION_PROMPT = """
You are a critical reflection agent.

Your task is to analyze a question-answering attempt
performed by a ReAct agent.

You will receive:
- The original question
- The agent's answer
- The tool calls and retrieved observations

Instructions:
1. Check whether the answer is supported by the observations.
2. Identify missing evidence or reasoning gaps.
3. Identify unsuccessful or irrelevant searches.
4. Suggest specific searches for the next attempt.
5. Do not assume the answer is incorrect.
6. If the answer is well-supported, recommend keeping it.
7. Do not invent facts or claim external verification.
8. Do not provide the final answer.

Provide concise and actionable feedback.
"""


# ==========================================
# 1. GENERATE HYBRID REFLECTION
# ==========================================

def generate_hybrid_reflection(
    question,
    answer,
    trajectory,
    reflection_prompt_override=None
):

    llm = get_llm()

    if trajectory:
        evidence = "\n\n".join(
            (
                f"Action: {step['action']}\n"
                f"Arguments: {step['arguments']}\n"
                f"Observation: {step['observation']}"
            )
            for step in trajectory
        )
    else:
        evidence = "No tool calls were made."

    messages = [
        SystemMessage(
            content=reflection_prompt_override or HYBRID_REFLECTION_PROMPT
        ),
        HumanMessage(
            content=(
                f"Question:\n{question}\n\n"
                f"Agent answer:\n{answer}\n\n"
                f"Tool trajectory:\n{evidence}\n\n"
                "Reflection:"
            )
        )
    ]

    response = llm.invoke(messages)

    return str(response.content).strip()


# ==========================================
# 2. HYBRID AGENT
# ==========================================

def hybrid_answer(
    question,
    documents,
    verbose=False,
    return_details=False,
    tool_factory=None,
    react_prompt_override=None,
    reflection_prompt_override=None
):

    memory = []
    history = []
    all_trajectories = []

    # --------------------------------------
    # FIRST REACT ATTEMPT
    # --------------------------------------

    first_answer, first_trajectory = react_answer(
        question=question,
        documents=documents,
        verbose=verbose,
        return_trajectory=True,
        tool_factory=tool_factory,
        system_prompt_override=react_prompt_override
    )

    history.append({
        "attempt": 1,
        "answer": first_answer
    })

    all_trajectories.append(first_trajectory)

    if verbose:
        print("\n===== HYBRID: FIRST ANSWER =====")
        print(first_answer)

    # --------------------------------------
    # REFLECTION
    # --------------------------------------

    reflection = generate_hybrid_reflection(
        question=question,
        answer=first_answer,
        trajectory=first_trajectory,
        reflection_prompt_override=reflection_prompt_override
    )

    memory.append(reflection)

    if verbose:
        print("\n===== HYBRID: REFLECTION =====")
        print(reflection)

    # --------------------------------------
    # SECOND REACT ATTEMPT
    # --------------------------------------

    second_answer, second_trajectory = react_answer(
        question=question,
        documents=documents,
        verbose=verbose,
        memory=memory,
        return_trajectory=True,
        tool_factory=tool_factory,
        system_prompt_override=react_prompt_override
    )

    history.append({
        "attempt": 2,
        "answer": second_answer
    })

    all_trajectories.append(second_trajectory)

    if verbose:
        print("\n===== HYBRID: FINAL ANSWER =====")
        print(second_answer)

    # --------------------------------------
    # RESULTS
    # --------------------------------------

    result = {
        "answer": second_answer,
        "trials": 2,
        "history": history,
        "memory": memory,
        "trajectories": all_trajectories,
        "tool_calls": sum(
            len(trajectory)
            for trajectory in all_trajectories
        )
    }

    if return_details:
        return result

    return second_answer

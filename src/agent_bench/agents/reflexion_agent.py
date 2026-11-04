
from langchain_core.messages import SystemMessage, HumanMessage

from agent_bench.llm import get_llm
from agent_bench.agents.answer_format import ANSWER_FORMAT_INSTRUCTIONS


# ==========================================
# PROMPTS
# ==========================================

ANSWER_PROMPT = """
You are a question-answering assistant.

Answer the question using your own knowledge.

Rules:
- Be accurate.
- Do not invent facts.
- If you do not know the answer, return Unknown.
"""

REFLECTION_PROMPT = """
You are a critical self-reflection agent.

Your task is to review a previous answer to a question.

Instructions:
1. Check whether the previous answer is likely correct.
2. Identify possible factual errors or missing information.
3. Do not assume the previous answer is wrong.
4. If the previous answer appears correct, recommend keeping it.
5. If the answer is uncertain, explain what needs verification.
6. Do not invent facts.
7. Provide concise and actionable feedback.

Do not provide a final answer.
"""


# ==========================================
# 1. GENERATE ANSWER
# ==========================================

def generate_answer(question, memory=None, answer_prompt_override=None):

    llm = get_llm()

    system_prompt = (
        (answer_prompt_override or ANSWER_PROMPT)
        + "\n\n"
        + ANSWER_FORMAT_INSTRUCTIONS
    )

    if memory:
        system_prompt += (
            "\n\nLessons from previous attempts:\n"
            + "\n".join(memory)
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)

    return str(response.content).strip()


# ==========================================
# 2. GENERATE REFLECTION
# ==========================================

def generate_reflection(question, answer, reflection_prompt_override=None):

    llm = get_llm()

    messages = [
        SystemMessage(content=reflection_prompt_override or REFLECTION_PROMPT),
        HumanMessage(
            content=(
                f"Question:\n{question}\n\n"
                f"Previous answer:\n{answer}\n\n"
                "Analyze the previous answer and provide "
                "actionable feedback."
            )
        )
    ]

    response = llm.invoke(messages)

    return str(response.content).strip()


# ==========================================
# 3. REFLEXION AGENT
# ==========================================

def reflexion_answer(
    question,
    verbose=False,
    return_details=False,
    answer_prompt_override=None,
    reflection_prompt_override=None
):

    # First attempt
    first_answer = generate_answer(question, answer_prompt_override=answer_prompt_override)

    # Reflection
    reflection = generate_reflection(
        question=question,
        answer=first_answer,
        reflection_prompt_override=reflection_prompt_override
    )

    # Second attempt using reflection memory
    final_answer = generate_answer(
        question=question,
        memory=[reflection],
        answer_prompt_override=answer_prompt_override
    )

    history = [
        {
            "attempt": 1,
            "answer": first_answer
        },
        {
            "attempt": 2,
            "answer": final_answer
        }
    ]

    if verbose:
        print("\n===== FIRST ATTEMPT =====")
        print("Answer:", first_answer)

        print("\n===== REFLECTION =====")
        print(reflection)

        print("\n===== SECOND ATTEMPT =====")
        print("Answer:", final_answer)

    result = {
        "answer": final_answer,
        "trials": 2,
        "history": history,
        "memory": [reflection]
    }

    if return_details:
        return result

    return final_answer

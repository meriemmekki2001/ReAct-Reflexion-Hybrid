
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

from agent_bench.llm import get_llm
from agent_bench.agents.answer_format import ANSWER_FORMAT_INSTRUCTIONS

BASELINE_SYSTEM_PROMPT = """
You are a question-answering assistant.

Answer the user's question using your own knowledge.

Rules:
- Do not invent facts.
- If you do not know the answer, say Unknown.

""" + ANSWER_FORMAT_INSTRUCTIONS



def baseline_answer(question, verbose=False, system_prompt_override=None):

    llm = get_llm()

    messages = [
        SystemMessage(content=system_prompt_override or BASELINE_SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)

    answer = str(response.content).strip()

    if verbose:
        print("\nQuestion:", question)
        print("Baseline answer:", answer)

    return answer

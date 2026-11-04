
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)

from agent_bench.llm import get_llm
from agent_bench.tools.documents import DocumentEnvironment, create_tools
from agent_bench.agents.answer_format import ANSWER_FORMAT_INSTRUCTIONS


MAX_STEPS = 6


REACT_SYSTEM_PROMPT = """
You are a multi-hop question-answering agent.

You have access to two tools:
- search_documents: find relevant documents
- read_document: read a document by title

Your task is to answer questions using retrieved evidence.

Instructions:
1. Search for relevant evidence before answering.
2. Identify missing information.
3. Use additional searches to connect facts across documents.
4. Do not repeat unsuccessful queries.
5. Do not invent unsupported facts.
6. Give only the shortest possible final answer.

If the available evidence is insufficient, answer Unknown.
"""


def react_answer(
    question,
    documents,
    verbose=False,
    memory=None,
    return_trajectory=False,
    tool_factory=None,
    system_prompt_override=None
):
    # Initialize model and document environment
    llm = get_llm()

    # GSM8K supplies no documents: do not initialize BM25 on an empty corpus.
    # HotpotQA retains its original document retrieval behavior.
    environment = DocumentEnvironment(documents) if documents else None if documents else None
    if tool_factory is not None:
        tools = tool_factory(environment)
    else:
        if environment is None:
            raise ValueError("Document tools require documents; provide a tool_factory for GSM8K.")
        tools = create_tools(environment)

    tool_map = {
        tool.name: tool
        for tool in tools
    }

    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
    (system_prompt_override or REACT_SYSTEM_PROMPT)
    + "\n\n"
    + ANSWER_FORMAT_INSTRUCTIONS
)

    # Add reflection memory if available
    if memory:
        system_prompt += (
            "\n\nLessons from previous attempts:\n"
            + "\n".join(memory)
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    trajectory = []
    answer = "Unknown"

    for step in range(MAX_STEPS):

        if verbose:
            print(f"\n===== ReAct Step {step + 1} =====")

        # Ask the LLM what to do next
        response = llm_with_tools.invoke(messages)

        messages.append(response)

        # If no tool call, the agent has finished
        if not response.tool_calls:

            answer = str(response.content).strip()

            if verbose:
                print("Final answer:", answer)

            break

        # Execute tool calls
        for call in response.tool_calls:

            tool_name = call["name"]
            tool_args = call["args"]
            tool_call_id = call["id"]

            if tool_name not in tool_map:
                observation = "Unknown tool."
            else:
                try:
                    observation = tool_map[
                        tool_name
                    ].invoke(tool_args)
                except Exception as error:
                    observation = f"Tool error: {error}"

            # Record action and observation
            trajectory.append({
                "step": step + 1,
                "action": tool_name,
                "arguments": tool_args,
                "observation": str(observation)
            })

            # Return observation to LLM
            messages.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call_id
                )
            )

            if verbose:
                print("Action:", tool_name)
                print("Arguments:", tool_args)
                print("Observation:", observation)

    else:
        # Maximum steps reached
        messages.append(
    HumanMessage(
        content=(
            "You have reached the maximum number "
            "of tool-use steps. "
            "Use only the retrieved evidence. "
            "Return only the shortest possible answer "
            "entity, number, or yes/no. "
            "Do not include any explanation."
        )
    )
)

        response = llm.invoke(messages)
        answer = str(response.content).strip()

    if return_trajectory:
        return answer, trajectory

    return answer

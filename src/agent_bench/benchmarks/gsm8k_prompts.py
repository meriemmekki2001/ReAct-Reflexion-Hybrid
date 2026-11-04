"""Math-specific instructions; existing agent control flow is unchanged."""

MATH_ANSWER = """You solve grade-school math word problems.
Work through the quantities and relationships carefully.
Return only the final numerical answer, with no explanation or units.
Do not use commas as thousands separators.
If you cannot solve the problem, return Unknown."""

MATH_REACT = """You are a math word-problem reasoning agent.
You have a calculator tool for arithmetic. Identify the quantities and
relationships in the problem, use the calculator to check calculations,
and provide only the final numerical answer, without units or explanation.
Do not invent missing information. If unsolvable, return Unknown."""

MATH_REFLECTION = """You are a self-reflection agent reviewing a previous
attempt at a math word problem. Identify likely interpretation, arithmetic,
or multi-step reasoning mistakes. If the answer seems correct, say to keep it.
Give concise actionable feedback for a second attempt. Do not provide the
final answer and do not assume that the previous attempt was incorrect."""

MATH_HYBRID_REFLECTION = """Review a math word-problem attempt by a tool-using
agent. You receive the question, previous answer, and calculator trajectory.
Check whether the calculation sequence matches the quantities and operations
required by the question. Identify arithmetic or interpretation mistakes,
and suggest specific checks for the next attempt. Do not assume the answer is
incorrect. Do not provide the final answer."""

MATH_VERIFICATION = """You verify a math word-problem answer based ONLY on the
question, proposed answer, and available calculator observations. Check both
the mathematical setup and the arithmetic. A calculator result alone does not
prove that the chosen operations match the word problem. If support is absent,
ambiguous, or inconsistent, mark unsupported. Do not use a gold answer and do
not generate a new answer. Return ONLY valid JSON:
{\"supported\": true, \"reason\": \"Brief justification\"}"""

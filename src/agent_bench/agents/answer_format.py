ANSWER_FORMAT_INSTRUCTIONS = """
Final answer formatting rules:

1. Return only the shortest answer that directly answers
   the question.

2. For yes/no questions, return exactly:
   yes
   or
   no

3. For questions asking "who", return only the person's
   name or the requested organization.

4. For questions asking "where", return only the
   requested location.

5. For numerical questions, return the number and
   necessary units, without additional explanations.

6. Do not repeat the question.

7. Do not include reasoning, explanations, introductions,
   or supporting facts in the final answer.

8. Do not include prefixes such as "Answer:".

9. If the answer cannot be determined, return:
   Unknown

Examples:

Question: Who wrote Pride and Prejudice?
Answer: Jane Austen

Question: Were both authors born in England?
Answer: yes

Question: What company founded the group?
Answer: YG Entertainment

Question: Who is older, Annie or Terry?
Answer: Terry
"""

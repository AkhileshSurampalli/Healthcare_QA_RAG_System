import sys
sys.path.append(".")

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.retriever import load_vector_store
from src.agent import build_agent
from src.evaluate import QUESTIONS, GROUND_TRUTHS, run_ragas_metrics

REFUSAL_TEXT = "I don't have enough information in the document to answer this."

# Cases the RAGAS question set can't exercise: each of these only passes if the
# agent actually reasons (multiple searches, a calculation, or a refusal)
# instead of answering from a single retrieval.
BEHAVIOR_CASES = [
    {
        "question": (
            "A child with pneumonia weighs 12 kg. Using the amoxicillin dosing "
            "regimen in the guideline, what is their per-dose and daily dose in mg?"
        ),
        "expect_tools": {"search_clinical_guidelines": 1, "calculate_dose": 1},
    },
    {
        "question": "Compare how malaria and pneumonia are managed according to the guideline.",
        "expect_tools": {"search_clinical_guidelines": 2},
    },
    {
        "question": "What is the capital of France?",
        "expect_refusal": True,
    },
]


def run_agent_and_trace(agent, question: str):
    """
    Run the agent once and pull out:
    - the final answer text
    - the retrieved passages from every search_clinical_guidelines call (for Ragas)
    - a count of how many times each tool was called (for the behavior checks)
    """
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    contexts = []
    tool_calls = {}
    answer = ""

    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_calls[message.name] = tool_calls.get(message.name, 0) + 1
            if message.name == "search_clinical_guidelines":
                contexts.append(message.content)
        elif isinstance(message, AIMessage) and message.content:
            answer = message.content

    return answer, contexts, tool_calls


def run_agent_ragas(agent):
    """
    Layer 1: score the agent with the same Ragas metrics evaluate.py uses for the
    static chain, but on the agent's aggregated retrieved passages and final
    answer. Reuses the existing single-fact question/ground-truth set, so this
    mainly checks the agent hasn't regressed on the easy cases.
    """
    print("\nRunning RAGAS question set through the agent...")
    answers, contexts = [], []

    for i, question in enumerate(QUESTIONS):
        print(f"  Question {i+1}/{len(QUESTIONS)}: {question[:50]}...")
        answer, ctx, _ = run_agent_and_trace(agent, question)
        answers.append(answer)
        contexts.append(ctx if ctx else [""])

    return run_ragas_metrics(
        QUESTIONS, answers, contexts, GROUND_TRUTHS,
        label="RAGAS EVALUATION RESULTS (ReAct agent)"
    )


def run_behavior_checks(agent):
    """
    Layer 2: does the agent actually reason the way it's meant to - multiple
    searches for a comparative question, the calculator tool for dosing math,
    a refusal for an out-of-scope question. Ragas can't measure any of this
    because it only looks at the final answer and context, not tool-call behavior.
    """
    print("\n" + "="*50)
    print("AGENT BEHAVIOR CHECKS")
    print("="*50)

    all_passed = True
    for case in BEHAVIOR_CASES:
        question = case["question"]
        answer, _, tool_calls = run_agent_and_trace(agent, question)

        failures = []
        if case.get("expect_refusal") and REFUSAL_TEXT not in answer:
            failures.append("expected refusal text, got a substantive answer")
        for tool_name, min_calls in case.get("expect_tools", {}).items():
            actual = tool_calls.get(tool_name, 0)
            if actual < min_calls:
                failures.append(f"expected >= {min_calls} call(s) to {tool_name}, got {actual}")

        status = "PASS" if not failures else "FAIL"
        all_passed = all_passed and not failures
        print(f"\n[{status}] {question[:70]}")
        print(f"  tool calls: {tool_calls}")
        for f in failures:
            print(f"  - {f}")

    print("\n" + "="*50)
    print("ALL BEHAVIOR CHECKS PASSED" if all_passed else "SOME BEHAVIOR CHECKS FAILED")
    print("="*50)
    return all_passed


if __name__ == "__main__":
    print("Loading vector store and building ReAct agent...")
    vector_store = load_vector_store()
    agent = build_agent(vector_store)

    run_agent_ragas(agent)
    run_behavior_checks(agent)

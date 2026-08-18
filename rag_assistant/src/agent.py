import os
import sys
sys.path.append(".")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.tools import build_tools

load_dotenv()

SYSTEM_PROMPT = """You are a clinical reasoning assistant that analyzes healthcare \
questions using ONLY the ingested clinical guideline document plus exact arithmetic.

Reason step by step (Thought -> Action -> Observation) before answering:
- Use the `search_clinical_guidelines` tool to find grounded facts. Break multi-part \
or comparative questions into separate, specific search queries and call the tool \
once per sub-question instead of guessing.
- Use the `calculate_dose` tool for any weight-based dosing arithmetic instead of \
computing it yourself.
- Never rely on outside/training knowledge for clinical facts. If the guideline does \
not contain the answer after searching, say exactly:
"I don't have enough information in the document to answer this."
- Cite which retrieved passages you relied on in your final answer.
- This assistant analyzes a reference document for educational purposes; it is not \
medical advice for real patients.
"""


def build_agent(vector_store, model: str = "gpt-4o-mini"):
    """
    Build a ReAct agent that reasons over the clinical guideline document instead
    of performing a single static retrieve -> augment -> generate pass.

    On each turn the agent decides whether to search, what to search for, whether
    a second (or third) search is needed, and whether to run the dose calculator,
    interleaving those tool calls with its own reasoning until it has enough
    grounded information to answer.
    """
    tools = build_tools(vector_store)
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    return create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def ask(agent, question: str):
    """Run the agent on a question, printing the full Thought/Action/Observation trace."""
    print(f"\n{'='*70}\nQuestion: {question}\n{'='*70}")

    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    final_answer = ""
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                print(f"\n[Action] {call['name']}({args})")
        elif isinstance(message, ToolMessage):
            print(f"[Observation] {message.content}")
        elif isinstance(message, AIMessage) and message.content:
            print(f"\n[Answer] {message.content}")
            final_answer = message.content

    return final_answer


if __name__ == "__main__":
    from src.retriever import load_vector_store

    print("Loading vector store...")
    vector_store = load_vector_store()

    print("Building ReAct healthcare agent...")
    agent = build_agent(vector_store)

    if len(sys.argv) > 1:
        ask(agent, " ".join(sys.argv[1:]))
    else:
        print("\nAgent ready. Running example questions...\n")

        ask(agent, "What are the symptoms of malaria and how is it treated?")
        ask(agent, "Compare how malaria and pneumonia are managed according to the guideline.")
        ask(agent, "A child with pneumonia weighs 12 kg. Using the amoxicillin dosing "
                    "regimen in the guideline, what is their per-dose and daily dose in mg?")
        ask(agent, "What is the capital of France?")

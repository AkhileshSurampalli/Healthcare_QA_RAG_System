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
questions using ONLY the ingested sources plus exact arithmetic. Those sources are:
1. A clinical guideline PDF and reference web pages (malaria, pneumonia, diabetes) -
   unstructured text, searched via `search_clinical_guidelines`.
2. A structured dosing table - exact drug/indication records, looked up via
   `lookup_dosing_table`. Prefer this over `search_clinical_guidelines` whenever the
   question needs a specific dosing number; a structured lookup can't return the
   wrong drug's numbers the way a text search sometimes can.

Reason step by step (Thought -> Action -> Observation) before answering:
- Break multi-part or comparative questions into separate, specific tool calls -
  one search per sub-question - instead of guessing from a single call.
- For dosing questions: call `lookup_dosing_table` first. If it returns a numeric
  mg/kg dose, follow up with `calculate_dose` for the patient's actual weight
  instead of doing the arithmetic yourself. If the drug isn't in the table, fall
  back to `search_clinical_guidelines`.
- Never rely on outside/training knowledge for clinical facts. If none of the tools
  turn up an answer, say exactly:
"I don't have enough information in the document to answer this."
- Cite which source each fact came from (the PDF, which web page, or the
  structured dosing table) in your final answer.
- This assistant analyzes reference sources for educational purposes; it is not \
medical advice for real patients. Any dosing table result is sample data and must \
be flagged as such, not presented as verified clinical guidance.
"""


def build_agent(vector_store, model: str = "gpt-4o-mini"):
    """
    Build a ReAct agent that reasons over multiple sources - the guideline PDF,
    reference web pages, and a structured dosing table - instead of performing a
    single static retrieve -> augment -> generate pass over one document.

    On each turn the agent decides whether to search unstructured text, look up a
    structured dosing record, run the dose calculator, or some combination across
    turns, interleaving those tool calls with its own reasoning until it has
    enough grounded information to answer.
    """
    tools = build_tools(vector_store)
    llm = ChatOpenAI(
        model=model,
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    return create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def run_with_trace(agent, question: str) -> dict:
    """
    Run the agent once and return the final answer plus a structured
    Thought/Action/Observation trace. Shared by the CLI (`ask`, which prints it)
    and the API (`api.py`, which returns it as JSON), so both surface the same
    reasoning steps instead of just the final answer.
    """
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    trace = []
    answer = ""
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            # Agent decided to use a tool
            for call in message.tool_calls:
                trace.append({"type": "action", "tool": call["name"], "args": call["args"]})
        elif isinstance(message, ToolMessage):
            # Tool returns the result
            trace.append({"type": "observation", "tool": message.name, "content": message.content})
        elif isinstance(message, AIMessage) and message.content:
            # AI's reasoning or final answer
            trace.append({"type": "answer", "content": message.content})
            answer = message.content

    return {"answer": answer, "trace": trace}


def ask(agent, question: str):
    """Run the agent on a question, printing the full Thought/Action/Observation trace."""
    print(f"\n{'='*70}\nQuestion: {question}\n{'='*70}")

    result = run_with_trace(agent, question)

    for step in result["trace"]:
        if step["type"] == "action":
            args = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
            print(f"\n[Action] {step['tool']}({args})")
        elif step["type"] == "observation":
            print(f"[Observation] {step['content']}")
        elif step["type"] == "answer":
            print(f"\n[Answer] {step['content']}")

    return result["answer"]


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
        ask(agent, "Compare how malaria and pneumonia are managed according to the sources.")
        ask(agent, "A child with pneumonia weighs 12 kg. What is the amoxicillin per-dose "
                    "and daily dose in mg?")
        ask(agent, "What is the dosing for artemether-lumefantrine and can I calculate "
                    "a per-kg dose for it?")
        ask(agent, "What is the capital of France?")

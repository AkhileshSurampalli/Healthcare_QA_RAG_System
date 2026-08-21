import os
import sys
sys.path.append(".")

# Modern Ragas and OpenAI imports
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas import EvaluationDataset, evaluate


from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

from dotenv import load_dotenv
from datasets import Dataset

from src.retriever import load_vector_store
from src.chain import build_rag_chain

load_dotenv()

# Initialize native OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create modern Ragas factories
evaluator_llm = llm_factory('gpt-4o-mini', client=openai_client)
evaluator_embeddings = embedding_factory('openai', model='text-embedding-3-small', client=openai_client)

QUESTIONS = [
    "What are the symptoms of malaria?",
    "How should diabetes be managed?",
    "What is the recommended treatment for pneumonia?",
    "What are the signs of severe malaria?",
    "What is the first line treatment for severe pneumonia?",
    "What dietary modifications are recommended for diabetes?",
    "How is Type 2 diabetes primarily managed?",
    "What antibiotics are used for very severe pneumonia?",
    "What are the features of uncomplicated malaria?",
    "What are the complications of malaria?"
]

GROUND_TRUTHS = [
    "Malaria symptoms include fever, chills, rigors, sweating, malaise, headache, myalgia, joint pains, nausea, vomiting, abdominal discomfort, and diarrhea.",
    "Diabetes management aims to abolish symptoms, correct hyperglycaemia, prevent complications, and modify diet with a nutritionist.",
    "Pneumonia is treated with amoxicillin if previously had cotrimoxazole, otherwise cotrimoxazole is given.",
    "Severe malaria presents with severe features requiring immediate medical attention affecting multiple organ systems.",
    "Severe pneumonia is treated with benzyl penicillin alone.",
    "Dietary modification is important in both Type 1 and Type 2 diabetes and must be individualized with a nutritionist.",
    "Type 2 diabetes is primarily managed through diet manipulation and exercise.",
    "Very severe pneumonia uses benzyl penicillin as first line antibiotic.",
    "Uncomplicated malaria classically presents with paroxysms of fever, chills, rigors, and sweating.",
    "Malaria can lead to severe complications including organ damage if untreated."
]


def run_ragas_metrics(questions, answers, contexts, ground_truths, label="RAGAS EVALUATION RESULTS"):
    """
    Score (question, answer, retrieved_contexts, ground_truth) rows with Ragas and
    print the averaged metrics. Shared by the static-chain and ReAct-agent evaluators
    so both are scored the same way.
    """
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }

    # Use the HuggingFace bypass for the Ragas dictionary bug
    hf_dataset = Dataset.from_dict(data)
    dataset = EvaluationDataset.from_hf_dataset(hf_dataset)

    # We pass the metrics empty, and let evaluate() inject the LLMs for us!
    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall()
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print("\n" + "="*50)
    print(label)
    print("="*50)
    df = results.to_pandas()
    print(f"Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
    print(f"Context Precision: {df['context_precision'].mean():.3f}")
    print(f"Context Recall:    {df['context_recall'].mean():.3f}")
    print("="*50)

    return results


def run_evaluation():
    print("\nLoading vector store and building RAG chain...")
    vector_store = load_vector_store()
    chain, retriever = build_rag_chain(vector_store)

    print("\nRunning questions through RAG chain...")
    answers = []
    contexts = []

    for i, question in enumerate(QUESTIONS):
        print(f"  Question {i+1}/{len(QUESTIONS)}: {question[:50]}...")
        answer = chain.invoke(question)
        answers.append(answer)
        docs = retriever.invoke(question)
        contexts.append([doc.page_content for doc in docs])

    return run_ragas_metrics(
        QUESTIONS, answers, contexts, GROUND_TRUTHS,
        label="RAGAS EVALUATION RESULTS (static chain)"
    )

if __name__ == "__main__":
    run_evaluation()
import os
import sys
sys.path.append(".")

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from src.retriever import load_vector_store
from src.chain import build_rag_chain

load_dotenv()

def build_eval_dataset(chain, retriever, questions, ground_truths):
    """
    Run each question through RAG chain and collect:
    - The generated answer
    - The retrieved context chunks
    - The ground truth answer you wrote manually

    RAGAS uses all gour to score the system
    
    """

    answers = []
    contexts = []

    for i, question in enumerate(questions):
        print(f"Running question {i+1/{len(questions)}: {questions[:50]}}...")

        # Get answer from chain
        answer = chain.invoke(question)
        answers.append(answer)

        # Get source chunks from retriever
        docs = retriever.invoke(question)
        contexts.append([doc.page_content for doc in docs])
    
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "context": contexts,
        "ground_truth": ground_truths
    })


def run_evaluation():
    """
    Main evaluation function.

    The ground_truths are answers we write manually
    based on what we know is in the document.
    These are the reference answers RAGAS compares against.
    """

    ## Test questions
    questions = [
        "What are the symptoms of malaria?",
        "How should diabetes be managed?",
        "What is the recommended treatment for pneumonia?",
        "What are the signs of severe malaria?"
        "What is the first line treatment for severe pneumonia?",
        "What dietary modifications are recommended for diabetes?",
        "What are the complications of malaria?",
        "How is Type 2 diabetes primarily managed?",
        "What antibiotics are used for very severe pneumonia?",
        "What are the features of uncomplicated malaria?"
    ]

    # Ground truth answers and they need not to be perfect
    ground_truths = [
        "Malaria symptoms include fever, chills, rigors, sweating, malaise, headache, myalgia, joint pains, nausea, vomiting, abdominal discomfort, and diarrhea.",
        "Diabetes management aims to abolish symptoms, correct hyperglycaemia, prevent complications, and modify diet with consultation from a nutritionist.",
        "Pneumonia is treated with amoxicillin if previously had cotrimoxazole, otherwise cotrimoxazole is given.",
        "Severe malaria presents with severe features requiring immediate medical attention including complications affecting multiple organ systems.",
        "Severe pneumonia is treated with benzyl penicillin alone.",
        "Dietary modification is important in both Type 1 and Type 2 diabetes and must be individualized with consultation from a nutritionist.",
        "Malaria can lead to severe complications including organ damage if untreated.",
        "Type 2 diabetes is primarily managed through diet manipulation and exercise.",
        "Very severe pneumonia uses benzyl penicillin as first line antibiotic.",
        "Uncomplicated malaria classically presents with paroxysms of fever, chills, rigors, and sweating."
    ]

    print("\nLoading vector store and building RAG chain...")
    vector_store = load_vector_store()
    chain, retriever = build_rag_chain(vector_store)

    print("\nRunning questions through RAG chain...")
    dataset = build_eval_dataset(chain, retriever, questions, ground_truths)

    print("\nRunning RAGAS evaluation - this calls OpenAI API for scoring...")
    scores = evaluate(dataset,
                      metrics=[faithfulness,
                               answer_relevancy,
                               context_precision,
                               context_recall])
    
    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    print(f"Faithfulness:      {scores['faithfulness']:.3f}")
    print(f"Answer Relevancy:  {scores['answer_relevancy']:.3f}")
    print(f"Context Precision: {scores['context_precision']:.3f}")
    print(f"Context Recall:    {scores['context_recall']:.3f}")
    print("="*50)
    print("\nWhat these mean:")
    print(f"  Faithfulness {scores['faithfulness']:.3f} — answers grounded in retrieved context")
    print(f"  Answer Relevancy {scores['answer_relevancy']:.3f} — answers address the questions")
    print(f"  Context Precision {scores['context_precision']:.3f} — retrieved chunks are relevant")
    print(f"  Context Recall {scores['context_recall']:.3f} — retrieval found needed information")

    return scores


if __name__ == "__main__":
    run_evaluation()

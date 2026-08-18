from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
import math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TEST_SET_PATH, OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
                    OPENROUTER_MODEL)


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _empty_evaluation() -> dict:
    """Return the stable result schema used by offline/error fallbacks."""
    return {**{metric: 0.0 for metric in METRIC_NAMES}, "per_question": []}


def _safe_score(value) -> float:
    """Convert RAGAS/NumPy values to finite Python floats."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if not OPENROUTER_API_KEY:
        print("  ⚠️  RAGAS evaluation skipped: OPENROUTER_API_KEY is not set")
        return _empty_evaluation()

    try:
        from datasets import Dataset
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        judge = ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0,
            default_headers={"X-OpenRouter-Title": "Lab 18 Production RAG"},
        )
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy,
                     context_precision, context_recall],
            llm=LangchainLLMWrapper(judge),
            embeddings=embeddings,
            raise_exceptions=False,
        )
        dataframe = result.to_pandas()
        per_question = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=_safe_score(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_score(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_score(row.get("context_precision", 0.0)),
                context_recall=_safe_score(row.get("context_recall", 0.0)),
            )
            for _, row in dataframe.iterrows()
        ]
        aggregate = {
            metric: (
                sum(getattr(item, metric) for item in per_question) / len(per_question)
                if per_question else 0.0
            )
            for metric in METRIC_NAMES
        }
        return {**aggregate, "per_question": per_question}
    except Exception as error:
        print(f"  ⚠️  RAGAS evaluation failed: {error}")
        return _empty_evaluation()


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "Answer contains claims unsupported by the retrieved context",
            "Tighten the grounded-answer prompt and lower model temperature",
        ),
        "answer_relevancy": (
            "Answer does not directly address the question",
            "Improve the answer prompt and preserve the user's query intent",
        ),
        "context_precision": (
            "Retrieved context contains too many irrelevant chunks",
            "Improve reranking or add source/version metadata filters",
        ),
        "context_recall": (
            "Retrieved context is missing information required for the answer",
            "Improve chunking, hybrid retrieval, or retrieve more candidates",
        ),
    }

    ranked = []
    for item in eval_results:
        scores = {metric: _safe_score(getattr(item, metric)) for metric in METRIC_NAMES}
        average = sum(scores.values()) / len(scores)
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        ranked.append((average, {
            "question": item.question,
            "expected": item.ground_truth,
            "got": item.answer,
            "contexts": item.contexts,
            "average_score": average,
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        }))

    ranked.sort(key=lambda entry: entry[0])
    return [failure for _, failure in ranked[:bottom_n]]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

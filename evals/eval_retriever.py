import os
# DEEPEVAL IMPORT HONE SE PEHLE CACHE DISABLE KARNA ZAROORI HAI
os.environ["DEEPEVAL_DISABLE_CACHE"] = "true"

import json
from pathlib import Path
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import OpenAIModel
from src.retriever import build_retriever

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_PATH = BASE_DIR / "goldens" / "retriever_goldens.json"
JUDGE_MODEL = "openai/gpt-oss-120b"
THRESHOLD = 0.7

def load_goldens(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)

def summarize_by_metric(result):
    scores = {}
    for test_result in getattr(result, "test_results", []):
        for metric in getattr(test_result, "metrics_data", []):
            score = getattr(metric, "score", None)
            name = getattr(metric, "name", None)
            if name is not None and score is not None:
                scores.setdefault(name, []).append(score)
    return {name: sum(vals) / len(vals) for name, vals in scores.items()}

def run(retriever):
    goldens = load_goldens(GOLDEN_PATH)
    test_cases = []
    
    # Sirf pehle 3 test cases taaki rate limit hit na ho
    for g in goldens[:3]:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]
        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated)"
            )
        )

    custom_judge = OpenAIModel(
        model=JUDGE_MODEL,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY")                     
    )

    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=custom_judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=custom_judge, include_reason=True),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "judge_model": JUDGE_MODEL,
        },
    )
    return summarize_by_metric(result)

if __name__ == "__main__":
    summary = run(build_retriever())
    for metric, score in summary.items():
        print(f"  {metric}: {score:.3f}")
# eval_retriever.py
import json
from pathlib import Path

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import OpenAIModel  # <-- Naya import add kiya
from src.retriever import build_retriever

load_dotenv()

# Path ko dynamic banaya taaki FileNotFoundError na aaye
BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_PATH = BASE_DIR / "goldens" / "retriever_goldens.json"
JUDGE_MODEL = "qwen2.5:0.5b"
THRESHOLD = 0.7


def load_goldens(path):
    """Load the human-authored golden cases from a JSON file."""
    with Path(path).open(encoding="utf-8") as file:
        goldens = json.load(file)
    if not isinstance(goldens, list):
        raise ValueError(f"Expected a list of goldens in {path}")
    return goldens


def summarize_by_metric(result):
    """Return the average score for each metric in an evaluation result."""
    scores = {}
    for test_result in getattr(result, "test_results", []):
        for metric in getattr(test_result, "metrics_data", []):
            score = getattr(metric, "score", None)
            name = getattr(metric, "name", None)
            if name is not None and score is not None:
                scores.setdefault(name, []).append(score)
    return {
        name: sum(metric_scores) / len(metric_scores)
        for name, metric_scores in scores.items()
    }


def print_summary(name, summary):
    print(f"{name} evaluation summary:")
    if not summary:
        print("No metric scores available.")
        return
    for metric, score in summary.items():
        print(f"  {metric}: {score:.3f}")


def run(retriever):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # ---> YAHAN OLLAMA KO CONFIGURE KIYA <---
    custom_judge = OpenAIModel(
    model=JUDGE_MODEL,
    base_url="http://localhost:11434/v1",
    api_key="ollama"
    )

    # 3. THE METRICS --- pass the custom local judge
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=custom_judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=custom_judge, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        run_async=False,  
        max_concurrent=1,
        hyperparameters={
            "retriever": "base_retriever",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 750,
            "chunk_overlap": 100,
            "top_k": 5,
            "judge_model": JUDGE_MODEL,
            "golden_set": str(GOLDEN_PATH),
        },
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(build_retriever())


if __name__ == "__main__":
    print_summary("retriever", run_local())
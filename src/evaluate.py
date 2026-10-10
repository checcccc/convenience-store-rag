"""评估：Recall@k 与 MRR。"""

from src.retrieval import search


def evaluate(index, client, queries: list[dict], k: int = 5) -> dict:
    """对评测集计算 Recall@k 与 MRR。"""
    recalls = []
    mrrs = []
    for q in queries:
        ids = [p.id for p in search(q["query"], index, client, top_n=k)]
        expected = q["expected_ids"]
        hit = any(e in ids for e in expected)
        recalls.append(1.0 if hit else 0.0)
        rank = next((i for i, pid in enumerate(ids, start=1) if pid in expected), None)
        mrrs.append(1.0 / rank if rank else 0.0)
    n = len(queries) or 1
    return {
        "recall_at_k": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
    }

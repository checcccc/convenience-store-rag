import types

from src.evaluate import evaluate
from src.ingestion import build_index


def _empty_client():
    def create(**kwargs):
        msg = types.SimpleNamespace(content="{}")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


def test_evaluate_perfect_recall(sample_products, fake_embedder):
    index = build_index(sample_products, fake_embedder)
    queries = [{"query": "零度可乐", "expected_ids": ["SKU-0001"]}]
    result = evaluate(index, _empty_client(), queries, k=5)
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] == 1.0

from src.ingestion import build_searchable_text
from src.rerank import rerank


def test_rerank_returns_subset(sample_products):
    class FakeReranker:
        def predict(self, pairs):
            return [float(len(pair[1])) for pair in pairs]

    candidates = sample_products[:3]
    result = rerank("查询", candidates, FakeReranker())
    assert len(result) <= len(candidates)
    lengths = [len(build_searchable_text(p)) for p in result]
    assert lengths == sorted(lengths, reverse=True)

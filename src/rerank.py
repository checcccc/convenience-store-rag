"""重排层：用交叉编码器对候选二次精排（可选）。"""

from src.data_model import Product
from src.ingestion import build_searchable_text


def rerank(query: str, candidates: list[Product], reranker) -> list[Product]:
    """用交叉编码器对 (query, searchable_text) 打分，返回分数降序的候选。"""
    if not candidates:
        return []
    pairs = [[query, build_searchable_text(p)] for p in candidates]
    scores = reranker.predict(pairs)
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [candidates[i] for i in order]

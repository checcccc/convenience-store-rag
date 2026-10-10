"""检索层：结构化过滤、向量/BM25 召回、RRF 融合。"""

from dataclasses import dataclass

from src.data_model import Product


@dataclass
class FilterConstraints:
    category_l1: str | None = None
    category_l2: str | None = None
    brand: str | None = None
    price_min: float | None = None
    price_max: float | None = None


def filter_products(products: list[Product], constraints: FilterConstraints) -> list[Product]:
    """按非空约束做等值/区间过滤；约束全空时返回原列表（顺序不变）。"""
    result = list(products)
    if constraints.category_l1 is not None:
        result = [p for p in result if p.category_l1 == constraints.category_l1]
    if constraints.category_l2 is not None:
        result = [p for p in result if p.category_l2 == constraints.category_l2]
    if constraints.brand is not None:
        result = [p for p in result if p.brand == constraints.brand]
    if constraints.price_min is not None:
        result = [p for p in result if p.price >= constraints.price_min]
    if constraints.price_max is not None:
        result = [p for p in result if p.price <= constraints.price_max]
    return result


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """倒数排名融合：按各路排名合成一份去重排序，比分相同按 id 升序稳定排序。"""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: (-scores[d], d))

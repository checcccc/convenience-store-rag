"""检索层：查询解析、结构化过滤、向量/BM25 召回、RRF 融合、编排。"""

import json
from dataclasses import dataclass

from src.data_model import Product
from src.ingestion import tokenize


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


def bm25_search(query: str, index, k: int = 30) -> list[str]:
    """BM25 字面召回：返回 BM25 分降序的商品 id 列表。"""
    tokens = tokenize(query)
    scores = index.bm25.get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [index.products[i].id for i in order[:k]]


def vector_search(query: str, index, k: int = 30) -> list[str]:
    """向量召回：返回余弦相似度降序的商品 id 列表。"""
    q = index.embedder.encode([query])
    n = min(k, index.collection.count())
    res = index.collection.query(query_embeddings=q, n_results=n)
    return res["ids"][0]


def extract_constraints(query: str, client) -> FilterConstraints:
    """用 LLM 从 query 抽取筛选条件；任何异常回退为空约束（全量召回）。"""
    system = (
        "从用户需求中抽取商品筛选条件，只输出 JSON，字段可为 null："
        "category_l1（大类）、category_l2（小类）、brand（品牌）、"
        "price_min（最低价，数字）、price_max（最高价，数字）。"
    )
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        return FilterConstraints()

    if not isinstance(data, dict):
        return FilterConstraints()

    def _to_float(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return FilterConstraints(
        category_l1=data.get("category_l1") or None,
        category_l2=data.get("category_l2") or None,
        brand=data.get("brand") or None,
        price_min=_to_float(data.get("price_min")),
        price_max=_to_float(data.get("price_max")),
    )


def _validate_constraints(constraints: FilterConstraints, products: list[Product]) -> FilterConstraints:
    """把不在数据中的 category/brand 约束置空，避免 LLM 误抽导致空池。"""
    l1 = {p.category_l1 for p in products}
    l2 = {p.category_l2 for p in products}
    brands = {p.brand for p in products}
    return FilterConstraints(
        category_l1=constraints.category_l1 if constraints.category_l1 in l1 else None,
        category_l2=constraints.category_l2 if constraints.category_l2 in l2 else None,
        brand=constraints.brand if constraints.brand in brands else None,
        price_min=constraints.price_min,
        price_max=constraints.price_max,
    )


def search(query: str, index, client, top_n: int = 5) -> list[Product]:
    """filter-then-rank 编排：空白 query 返回空；硬过滤 → 池内向量+BM25 → RRF → top_n。"""
    if not query.strip():
        return []
    constraints = _validate_constraints(extract_constraints(query, client), index.products)
    pool = filter_products(index.products, constraints)
    if not pool:
        pool = index.products  # 约束过严导致空池：兜底全量召回
    pool_ids = {p.id for p in pool}
    full = len(index.products)
    vector_ids = [i for i in vector_search(query, index, k=full) if i in pool_ids][:30]
    bm25_ids = [i for i in bm25_search(query, index, k=full) if i in pool_ids][:30]
    fused = rrf_fuse([vector_ids, bm25_ids])
    by_id = {p.id: p for p in index.products}
    return [by_id[i] for i in fused[:top_n]]

import types

from rank_bm25 import BM25Okapi

from src.ingestion import build_index
from src.retrieval import (
    FilterConstraints,
    bm25_search,
    extract_constraints,
    filter_products,
    rrf_fuse,
    search,
    vector_search,
)


def _fake_client(content: str):
    msg = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=msg)
    resp = types.SimpleNamespace(choices=[choice])
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kw: resp)
        )
    )


def test_filter_by_category(sample_products):
    constraints = FilterConstraints(category_l1="饮料")
    result = filter_products(sample_products, constraints)
    assert len(result) > 0
    assert all(p.category_l1 == "饮料" for p in result)


def test_filter_by_price_range(sample_products):
    constraints = FilterConstraints(price_min=3.0, price_max=5.0)
    result = filter_products(sample_products, constraints)
    assert len(result) > 0
    assert all(3.0 <= p.price <= 5.0 for p in result)


def test_filter_no_constraints_returns_all(sample_products):
    result = filter_products(sample_products, FilterConstraints())
    assert result == sample_products


def test_rrf_fuse_ranking():
    result = rrf_fuse([["a", "b", "c"], ["b", "a"]], k=60)
    assert result == ["a", "b", "c"]


def test_rrf_fuse_dedupes():
    result = rrf_fuse([["a", "b"], ["a", "c"]])
    assert len(result) == 3
    assert result[0] == "a"
    assert len(set(result)) == 3


def test_bm25_search_ranking(sample_products):
    corpus = [
        ["可乐", "饮料"] if "可乐" in p.name else ["其他", "商品"]
        for p in sample_products
    ]
    index = types.SimpleNamespace(bm25=BM25Okapi(corpus), products=sample_products)
    result = bm25_search("可乐", index, k=30)
    assert set(result[:2]) == {"SKU-0001", "SKU-0002"}


def test_vector_search_ordering(sample_products, fake_embedder):
    index = build_index(sample_products, fake_embedder)
    result = vector_search("零度可乐", index, k=3)
    assert result[0] == "SKU-0001"


def test_extract_constraints_valid():
    client = _fake_client('{"category_l1": "饮料", "price_max": 5}')
    c = extract_constraints("5元以下的饮料", client)
    assert c.category_l1 == "饮料"
    assert c.price_max == 5.0
    assert c.category_l2 is None


def test_extract_constraints_malformed():
    client = _fake_client("这不是JSON")
    c = extract_constraints("随便", client)
    assert c == FilterConstraints()


def test_search_filter_then_rank(sample_products, fake_embedder):
    index = build_index(sample_products, fake_embedder)
    client = _fake_client('{"category_l1": "饮料"}')
    result = search("饮料", index, client, top_n=2)
    assert result
    assert len(result) <= 2
    assert all(p.category_l1 == "饮料" for p in result)


def test_search_empty_query(sample_products, fake_embedder):
    index = build_index(sample_products, fake_embedder)
    result = search("   ", index, None)
    assert result == []


def test_extract_constraints_valid_json_non_object():
    client = _fake_client("null")
    c = extract_constraints("随便", client)
    assert c == FilterConstraints()

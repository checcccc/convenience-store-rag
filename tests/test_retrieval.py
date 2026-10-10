from src.retrieval import FilterConstraints, filter_products, rrf_fuse


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

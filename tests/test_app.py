import types

from src.app import answer, build_client
from src.ingestion import build_index


def _make_client():
    def create(**kwargs):
        system = kwargs["messages"][0]["content"]
        if "筛选条件" in system:
            content = '{"category_l1": "饮料"}'
        else:
            content = '{"summary": "为你找到几款饮料", "products": []}'
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


def test_answer_end_to_end(sample_products, fake_embedder):
    index = build_index(sample_products, fake_embedder)
    result = answer("无糖饮料", index, _make_client())
    assert isinstance(result, dict)
    assert isinstance(result["summary"], str)
    assert result["summary"]
    assert isinstance(result["products"], list)
    assert result["products"]
    for p in result["products"]:
        assert {"id", "name", "price"} <= set(p)


def test_answer_empty_query(sample_products, fake_embedder):
    index = build_index(sample_products, fake_embedder)
    result = answer("   ", index, _make_client())
    assert result == {"summary": "没有找到相关商品", "products": []}


def test_build_client_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert build_client() is None

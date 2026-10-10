import types

from src.generation import build_messages, generate_summary, parse_output


def test_build_messages_contains_system_and_products(sample_products):
    messages = build_messages("无糖饮料", sample_products[:2])
    assert messages[0]["role"] == "system"
    assert "绝不能编造" in messages[0]["content"]
    user = messages[1]["content"]
    assert "无糖饮料" in user
    for p in sample_products[:2]:
        assert p.name in user


def test_parse_output_valid():
    text = '{"summary": "为你找到2款商品", "products": [{"id": "SKU-0001", "name": "零度可乐"}]}'
    result = parse_output(text)
    assert result.summary == "为你找到2款商品"
    assert result.products == [{"id": "SKU-0001", "name": "零度可乐"}]


def test_parse_output_malformed():
    result = parse_output("这不是JSON")
    assert result.summary == "未找到合适的商品"
    assert result.products == []


def test_generate_summary_no_products():
    result = generate_summary("x", [], None)
    assert result.summary == "没有找到相关商品"
    assert result.products == []


def test_generate_summary_parses_client_output(sample_products):
    msg = types.SimpleNamespace(content='{"summary": "为你找到2款商品", "products": []}')
    choice = types.SimpleNamespace(message=msg)
    resp = types.SimpleNamespace(choices=[choice])
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kw: resp)
        )
    )
    result = generate_summary("无糖", sample_products[:1], client)
    assert result.summary == "为你找到2款商品"


def test_parse_output_valid_json_non_object():
    for text in ["null", "[]", '"text"', "123"]:
        result = parse_output(text)
        assert result.summary == "未找到合适的商品"
        assert result.products == []


def test_generate_summary_client_error(sample_products):
    def create(**kwargs):
        raise RuntimeError("network down")

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        )
    )
    result = generate_summary("无糖", sample_products[:1], client)
    assert result.summary == "为你找到 1 款相关商品：零度可乐 330ml 罐装"
    assert result.products == []

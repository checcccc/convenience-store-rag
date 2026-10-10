"""生成层：DeepSeek 基于检索结果生成自然语言小结。"""

import json
from dataclasses import dataclass, field

from src.data_model import Product

SYSTEM_PROMPT = (
    "你是便利店商品助手。只能基于给定给你的商品作答，"
    "绝不能编造给定列表之外的商品或信息。请输出 JSON。"
)


@dataclass
class GenerationResult:
    summary: str
    products: list[dict] = field(default_factory=list)


def build_messages(query: str, products: list[Product]) -> list[dict]:
    """构造 system + user 消息；user 含用户 query 与候选商品 JSON。"""
    payload = [
        {
            "id": p.id,
            "name": p.name,
            "category": f"{p.category_l1}/{p.category_l2}",
            "brand": p.brand,
            "price": p.price,
            "tags": p.tags,
            "description": p.description,
        }
        for p in products
    ]
    user = f"用户需求：{query}\n候选商品：{json.dumps(payload, ensure_ascii=False)}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_output(text: str) -> GenerationResult:
    """解析 LLM 返回的 JSON；非法或缺 summary 时回退为确定性结果。"""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return GenerationResult(summary="未找到合适的商品", products=[])
    summary = data.get("summary")
    products = data.get("products", [])
    if not isinstance(summary, str) or not summary:
        return GenerationResult(summary="未找到合适的商品", products=[])
    if not isinstance(products, list):
        products = []
    return GenerationResult(summary=summary, products=products)


def generate_summary(query: str, products: list[Product], client) -> GenerationResult:
    """空候选不调 LLM 直接返回；否则调 client 生成并解析。"""
    if not products:
        return GenerationResult(summary="没有找到相关商品", products=[])
    messages = build_messages(query, products)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return parse_output(resp.choices[0].message.content)

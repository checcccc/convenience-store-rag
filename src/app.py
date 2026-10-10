"""端到端入口：组装 pipeline 并提供 CLI。"""

import json
import os
import sys
from pathlib import Path

from src.data_model import load_products
from src.generation import generate_summary
from src.ingestion import build_index
from src.retrieval import search

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "products.csv"


def build_embedder(model: str = "BAAI/bge-small-zh-v1.5"):
    """惰性加载本地 BGE 嵌入模型（重依赖，仅在真实运行时导入）。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model)


def build_client():
    """惰性构建国内 LLM 客户端（OpenAI 兼容 API）。"""
    from openai import OpenAI

    return OpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        api_key=os.environ.get("LLM_API_KEY", ""),
    )


def answer(query: str, index, client) -> dict:
    """串起 search → generate_summary，返回 {summary, products}。"""
    products = search(query, index, client, top_n=5)
    result = generate_summary(query, products, client)
    return {
        "summary": result.summary,
        "products": [
            {
                "id": p.id,
                "name": p.name,
                "category": f"{p.category_l1}/{p.category_l2}",
                "brand": p.brand,
                "price": p.price,
            }
            for p in products
        ],
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("用法：python -m src.app \"你的需求\"")
        sys.exit(1)
    query = sys.argv[1]
    products = load_products(DATA_PATH)
    embedder = build_embedder()
    client = build_client()
    index = build_index(products, embedder)
    result = answer(query, index, client)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

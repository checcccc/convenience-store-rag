"""数据接入：可检索文本构造、分词、索引构建。"""

import uuid
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

from src.data_model import Product


def build_searchable_text(p: Product) -> str:
    """把一条商品拼成可检索文本（向量与 BM25 共用同一段文本）。"""
    return (
        f"{p.name} {p.spec}，分类：{p.category_l1}/{p.category_l2}，品牌：{p.brand}，"
        f"特点：{'、'.join(p.tags)}，{p.description}"
    )


def tokenize(text: str) -> list[str]:
    """中文分词（jieba lcut）。"""
    return jieba.lcut(text)


@dataclass
class StoreIndex:
    products: list[Product]
    searchable_texts: list[str]
    tokenized_texts: list[list[str]]
    bm25: object
    collection: object
    embedder: object


def build_index(products: list[Product], embedder) -> StoreIndex:
    """离线建索引：可检索文本、分词、BM25 索引、Chroma 向量库（余弦距离）。"""
    import chromadb

    searchable_texts = [build_searchable_text(p) for p in products]
    tokenized_texts = [tokenize(t) for t in searchable_texts]
    bm25 = BM25Okapi(tokenized_texts)

    embeddings = embedder.encode(searchable_texts)
    client = chromadb.Client()
    collection = client.create_collection(
        name=f"products_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[p.id for p in products],
        embeddings=embeddings,
        documents=searchable_texts,
        metadatas=[
            {
                "name": p.name,
                "category_l1": p.category_l1,
                "category_l2": p.category_l2,
                "brand": p.brand,
                "price": p.price,
            }
            for p in products
        ],
    )
    return StoreIndex(
        products=products,
        searchable_texts=searchable_texts,
        tokenized_texts=tokenized_texts,
        bm25=bm25,
        collection=collection,
        embedder=embedder,
    )

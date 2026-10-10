"""数据接入：可检索文本构造、分词。"""

import jieba

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

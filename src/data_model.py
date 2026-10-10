"""商品数据模型：Product 定义与 CSV 加载/校验。"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

# 必填字段：除 tags、description 外均为必填。
REQUIRED_FIELDS = ["id", "name", "category_l1", "category_l2", "brand", "price", "spec"]


@dataclass
class Product:
    id: str
    name: str
    category_l1: str
    category_l2: str
    brand: str
    price: float
    spec: str
    tags: list[str] = field(default_factory=list)
    description: str = ""


def load_products(path: str | Path) -> list[Product]:
    """从 CSV 加载商品，校验必填字段与 price 数值。失败抛 ValueError（消息含行号或 id）。"""
    products: list[Product] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            for field_name in REQUIRED_FIELDS:
                if not row.get(field_name, "").strip():
                    raise ValueError(f"第 {line_no} 行缺少必填字段 {field_name}")
            try:
                price = float(row["price"])
            except ValueError:
                raise ValueError(f"商品 {row['id']} 的 price 非法: {row['price']}") from None
            products.append(
                Product(
                    id=row["id"].strip(),
                    name=row["name"].strip(),
                    category_l1=row["category_l1"].strip(),
                    category_l2=row["category_l2"].strip(),
                    brand=row["brand"].strip(),
                    price=price,
                    spec=row["spec"].strip(),
                    tags=[t.strip() for t in (row.get("tags") or "").split("|") if t.strip()],
                    description=(row.get("description") or "").strip(),
                )
            )
    return products

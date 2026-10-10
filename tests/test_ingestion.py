from src.data_model import Product
from src.ingestion import build_searchable_text, tokenize


def test_build_searchable_text():
    p = Product(
        id="SKU-0001",
        name="零度可乐",
        category_l1="饮料",
        category_l2="碳酸饮料",
        brand="可口可乐",
        price=3.5,
        spec="330ml",
        tags=["无糖", "无咖啡因", "低热量"],
        description="适合想提神又不想摄入糖分的人群",
    )
    assert build_searchable_text(p) == (
        "零度可乐 330ml，分类：饮料/碳酸饮料，品牌：可口可乐，"
        "特点：无糖、无咖啡因、低热量，适合想提神又不想摄入糖分的人群"
    )


def test_tokenize_chinese():
    tokens = tokenize("无糖可乐")
    assert len(tokens) > 0
    assert all(t.strip() for t in tokens)

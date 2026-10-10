import pytest

from src.data_model import Product, load_products

HEADER = "id,name,category_l1,category_l2,brand,price,spec,tags,description\n"


def _write_csv(tmp_path, content: str) -> str:
    path = tmp_path / "products.csv"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_load_products_valid(tmp_path):
    path = _write_csv(
        tmp_path,
        HEADER
        + "SKU-0001,零度可乐 330ml 罐装,饮料,碳酸饮料,可口可乐,3.5,330ml,无糖|无咖啡因|低热量,适合想提神又不想摄入糖分的人群\n"
        + "SKU-0002,薯片 原味,零食,膨化食品,乐事,8.9,90g,,\n",
    )
    products = load_products(path)
    assert len(products) == 2

    p = products[0]
    assert p.id == "SKU-0001"
    assert p.name == "零度可乐 330ml 罐装"
    assert p.category_l1 == "饮料"
    assert p.category_l2 == "碳酸饮料"
    assert p.brand == "可口可乐"
    assert p.price == 3.5
    assert isinstance(p.price, float)
    assert p.spec == "330ml"
    assert p.tags == ["无糖", "无咖啡因", "低热量"]
    assert isinstance(p.tags, list)
    assert p.description == "适合想提神又不想摄入糖分的人群"

    assert products[1].tags == []
    assert products[1].description == ""


def test_load_products_invalid_price(tmp_path):
    path = _write_csv(
        tmp_path,
        HEADER + "SKU-0001,零度可乐,饮料,碳酸饮料,可口可乐,abc,330ml,无糖,\n",
    )
    with pytest.raises(ValueError, match="SKU-0001"):
        load_products(path)


def test_load_products_missing_field(tmp_path):
    path = _write_csv(
        tmp_path,
        HEADER + "SKU-0001,零度可乐,饮料,碳酸饮料,,3.5,330ml,无糖,\n",
    )
    with pytest.raises(ValueError):
        load_products(path)

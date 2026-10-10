# 便利店 AI 助手（RAG）

一个 Python 实现的便利店商品检索助手：用户用自然语言描述需求（清楚或模糊），系统返回匹配的商品列表 + 一句自然语言小结。技术路线是 RAG（检索增强生成）——先检索，再让 LLM 基于检索结果组织回答，杜绝幻觉。

## 架构

5 层流水线 + 1 条离线管线：

```
query → ① 查询解析（LLM 抽结构化约束）
      → ② 检索（结构化硬过滤 → 向量召回 + BM25 召回 → RRF 融合）
      → ③ 重排（bge-reranker，可选）
      → ④ 生成（DeepSeek 只组织语言，钉死在检索结果上）
      → 输出：summary + 商品列表
```

离线管线（与查询分离）：`商品数据 CSV → 可检索文本构造 → BGE embedding / BM25 索引 / 结构化字段`。

完整设计见 `docs/superpowers/specs/`，实现计划见 `docs/superpowers/plans/`。

## 目录结构

```
data/products.csv        # 样例商品数据（tags 精标注）
src/
  data_model.py          # Product + 加载/校验
  ingestion.py           # 文本构造 + 分词 + 建索引（StoreIndex）
  retrieval.py           # 查询解析 / 过滤 / 向量 / BM25 / RRF / search 编排
  rerank.py              # 可选交叉编码器重排
  generation.py          # LLM 生成小结
  evaluate.py            # Recall@k / MRR
  app.py                 # CLI 入口
eval/queries.json        # 评测集
tests/                   # 单元测试（FakeEmbedder/FakeClient，无需模型与 key）
```

## 快速开始

```bash
pip install -r requirements.txt

# 运行 CLI（需先设置国内 LLM 环境变量，首次运行会下载 BGE 模型）
$env:LLM_BASE_URL = "https://api.deepseek.com"
$env:LLM_API_KEY  = "你的 key"
python -m src.app "无糖、不含咖啡因的饮料"
```

输出形如：

```json
{
  "summary": "为你找到 3 款无糖、不含咖啡因的饮料：……",
  "products": [
    {"id": "SKU-0001", "name": "零度可乐 330ml 罐装", "category": "饮料/碳酸饮料", "brand": "可口可乐", "price": 3.5}
  ]
}
```

## 测试与评估

```bash
pytest                 # 单元测试（全部用 Fake 替身，不依赖模型/网络/API key）
python -m src.evaluate # 未提供 CLI，评估入口见 tests/test_evaluate.py
```

模型选型：embedding `BAAI/bge-small-zh-v1.5`，rerank（可选）`BAAI/bge-reranker-base`，LLM 国内 OpenAI 兼容 API（DeepSeek/通义）。

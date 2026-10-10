# 便利店 AI 助手（RAG）设计文档

- 日期：2026-10-11
- 状态：已确认（下一步：写实现计划）
- 类型：学习 / Demo / 原型

## 1. 背景与目标

做一个便利店的 AI 助手，帮助用户在琳琅满目的商品中快速找到想要的那一个。
用户用自然语言描述需求（清楚的或模糊的），系统返回匹配的商品列表，并附一句自然语言小结。

技术上走 RAG（检索增强生成）：不靠模型硬背商品库，而是先"检索"出相关商品，再让模型基于检索结果组织回答。

## 2. 需求范围

**支持（In scope）：**

- 精确检索：用户描述清楚，找到具体某类商品（如"可口可乐""XX 牌的薯片"）。靠结构化字段 + 关键词。
- 语义 / 模糊搜索：用户说得模糊，靠语义匹配（如"提神醒脑的饮料""给小孩吃的健康零食"）。

**不支持（Out of scope，YAGNI）：**

- 个性化推荐（无用户画像 / 无协同过滤）。
- 长问答（不做成分、保质期等闲聊式多轮问答）。
- 实时库存 / 货架位置（Demo 阶段不接实时数据）。

## 3. 技术选型

| 项 | 选型 | 说明 |
|----|------|------|
| 语言 | Python | RAG 生态最成熟 |
| embedding | `sentence-transformers` 加载 `BAAI/bge-small-zh-v1.5` | 本地、免费、CPU 可跑、中文效果好；后续可无缝换 `bge-large-zh-v1.5` |
| 向量库 | Chroma | 本地、零配置 |
| 关键词检索 | `rank_bm25` + `jieba` 分词 | BM25 字面匹配 |
| 融合 | 自写 RRF（倒数排名融合） | 合并多路召回排名 |
| rerank（可选） | `BAAI/bge-reranker-base` | 二次精排，第一阶段不开启 |
| LLM | 国内 API（DeepSeek / 通义等），OpenAI 兼容 SDK | 生成自然语言小结 |
| 结构化过滤 | 内存 DataFrame / SQLite | Demo 规模，不上重数据库 |

## 4. 系统架构

5 层流水线 + 1 条离线管线：

```
用户 query
   │
   ▼
① 查询层 Query —— 拆成结构化约束 + 语义/字面意图
   ▼
② 检索层 Retrieval —— 硬过滤 → 向量召回 + BM25 召回 → RRF 融合
   ▼
③ 重排层 Rerank（可选）—— bge-reranker 二次精排
   ▼
④ 生成层 Generation —— DeepSeek 基于 top-N 生成一句小结
   ▼
输出：summary + 商品卡片列表
```

离线管线（与查询分离，跑一次）：

```
商品数据(CSV/JSON) → 文本构造 → embedding 建向量库 / 建 BM25 索引 / 结构化字段落库
```

### 模块划分（每模块一个职责、可独立测试）

| 模块 | 职责 | 依赖 |
|------|------|------|
| `data_model` | 商品 schema + 加载/校验 | 无 |
| `ingestion` | 文本构造 + embedding + 建三套索引 | `data_model` |
| `retrieval` | 结构化过滤 / 向量 / BM25 / RRF 融合（各一个函数） | `ingestion` 的索引 |
| `rerank` | 二次精排（可选） | `retrieval` |
| `generation` | LLM 生成小结 | `retrieval`/`rerank` |
| `app` | 组装 pipeline + CLI 入口 | 以上全部 |

## 5. 商品数据模型

| 字段 | 类型 | 例子 | 主要服务 |
|------|------|------|----------|
| `id` | string | `SKU-0001` | 唯一标识 |
| `name` | string | 零度可乐 330ml 罐装 | BM25 字面 |
| `category_l1` | string | 饮料 | 精确过滤 |
| `category_l2` | string | 碳酸饮料 | 精确过滤 |
| `brand` | string | 可口可乐 | 过滤 + 字面 |
| `price` | float | 3.5 | 价格区间过滤 |
| `spec` | string | 330ml | 字面补充 |
| `tags` | list[str] | `["无糖","无咖啡因","低热量"]` | **语义检索命脉** |
| `description` | string | 适合想提神又不想摄入糖分的人群 | 语义补充 |

**关键判断**：`tags` 是语义检索的命脉。模糊搜索问的多是属性（"无糖""低脂""适合儿童"），这些词必须被显式标注在数据里，向量才能匹配到。因此样例数据的标签质量比数据量更重要。

## 6. 可检索文本与 Ingestion 管线

### 6.1 可检索文本（chunk 构造）

向量和 BM25 吃"文本"不吃"表"。把一条商品拼成一段可检索文本，同一段文本喂给两套索引：

```
"{name} {spec}，分类：{category_l1}/{category_l2}，品牌：{brand}，
 特点：{tags 顿号连接}，{description}"
```

### 6.2 Ingestion 步骤

1. 加载 CSV/JSON + 校验字段完整。
2. 构造可检索文本。
3. 生成 BGE 向量 → 存 Chroma（结构化字段一并存进 metadata，避免回表）。
4. jieba 分词 → 建 BM25 索引。
5. 结构化字段 → 内存 DataFrame/SQLite 供过滤。

## 7. 检索层

### 7.1 查询解析

用 DeepSeek 做结构化抽取，输出 `{category, brand, price_min, price_max}`。抽取失败或为空 → 当作无约束 → 全量召回，不影响主流程。整句 query 原样用于向量和 BM25。

### 7.2 三条召回

- 结构化过滤：`category`/`brand` 等值匹配、`price` 区间过滤。约束为空 → 全量。
- 向量召回：query → BGE 向量 → Chroma 余弦相似度 top-k=30。
- BM25 召回：query → jieba 分词 → BM25 打分 top-k=30。

### 7.3 融合顺序：先过滤、后融合（filter-then-rank）

```
全量商品 → 硬过滤（有约束才过滤）→ 候选池
        → 池内 向量 + BM25 双路召回
        → RRF 融合 → top-20 →（可选 rerank）→ top-5~8
```

**RRF 公式**：`RRFscore(d) = Σ 1/(k + rank_i(d))`，k 取 60。用排名而非原始分数，解决向量相似度与 BM25 分数不可比的问题。

**理由**：简单直观；"用户明确说分类/价格"应是硬门槛而非加分项。风险（约束抽错漏召回）由"失败回退全量"兜底。将来更鲁棒可升级为"过滤作为一路召回参与 RRF"。

### 7.4 Rerank（可插拔，第一阶段关闭）

融合后 top-20 用 `bge-reranker-base` 二次精排，取 top-5~8。第一阶段不启用，设计成可插拔一步。

## 8. 生成层

DeepSeek 只做"把检索好的商品组织成人话"，不做检索、不凭空发挥。

- System prompt：你是便利店商品助手，只能基于给定商品作答，**绝不能编造给定列表之外的商品或信息**。
- User：用户原始 query + top-N 商品 JSON。
- 温度 ≈ 0.1，要求输出 JSON，保证稳定可解析。

输出格式：

```json
{
  "summary": "为你找到 3 款无糖、不含咖啡因的饮料：零度可乐最贴合，无糖气泡水更清爽……",
  "products": [
    {
      "id": "SKU-0001",
      "name": "零度可乐 330ml 罐装",
      "category": "饮料/碳酸饮料",
      "brand": "可口可乐",
      "price": 3.5,
      "match_reason": "无糖、无咖啡因"
    }
  ]
}
```

## 9. 模型选型汇总

| 角色 | 模型 |
|------|------|
| embedding | `BAAI/bge-small-zh-v1.5`（512 维，CPU 可跑；可换 large） |
| rerank（可选） | `BAAI/bge-reranker-base` |
| LLM | DeepSeek / 通义等国内 API |

## 10. 评估与测试

- 评测集：手写几十条 `(query, 期望命中商品 id 列表)`，覆盖精确 / 模糊 / 带价格约束三类。
- 指标：`Recall@k`、`MRR`。调参（top-k、RRF 的 k、是否开 rerank、换模型）时用数字判断，不用肉眼。
- 单元测试：数据校验、文本构造、RRF 融合排名、约束提取、输出 JSON 可解析。

## 11. 目录结构

```
convenience-store-rag/
├── data/products.csv            # 样例商品数据（精心标注 tags）
├── src/
│   ├── data_model.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── rerank.py                # 可选
│   ├── generation.py
│   └── app.py
├── eval/queries.json + evaluate.py
├── tests/test_*.py
├── requirements.txt
└── README.md
```

## 12. 里程碑（实施节奏）

1. **M1 最小链路**：样例数据 → 文本构造 → BGE embedding → Chroma 向量召回 → DeepSeek 生成。跑通"纯向量 + 生成"。
2. **M2 混合检索**：加入结构化过滤 + BM25 + RRF 融合，覆盖精确检索。
3. **M3 精排与评估**：接入可选 rerank；建评测集，用 Recall@k / MRR 量化效果。

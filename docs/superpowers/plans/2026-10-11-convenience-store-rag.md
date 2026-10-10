# 便利店 RAG 助手 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一个 Python 便利店商品检索助手：自然语言 query → 结构化过滤 + 向量/BM25 双路召回 → RRF 融合 → DeepSeek 生成一句小结 + 商品列表。

**Architecture:** 5 层流水线（查询解析 → 检索 → 可选重排 → 生成）+ 1 条离线管线（数据 → 文本构造 → embedding/BM25/结构化索引）。检索层采用 filter-then-rank：先硬过滤，再在候选池内做向量 + BM25 双路召回，RRF 融合。LLM 只做"组织语言"，钉死在检索结果上。

**Tech Stack:** Python 3.10+，`sentence-transformers`（BGE）、`chromadb`、`rank-bm25`、`jieba`、`openai`（DeepSeek 兼容 API）、`pytest`。

**Spec:** `docs/superpowers/specs/2026-10-11-convenience-store-rag-design.md`

## Global Constraints

- Python 3.10+；用 `dataclass` 定义数据类；类型注解用 `list[X]`、`X | None`（不 `from __future__`）。
- embedding 模型：`BAAI/bge-small-zh-v1.5`（512 维）。
- rerank 模型：`BAAI/bge-reranker-base`（M3 才用）。
- LLM：OpenAI 兼容 SDK；`base_url` 和 `api_key` 从环境变量 `LLM_BASE_URL`、`LLM_API_KEY` 读取。
- 检索常量：向量/BM25 召回 `k=30`，RRF `k=60`，融合后 top-20，最终 `top_n=5`，生成温度 `0.1`。
- 可检索文本模板：`"{name} {spec}，分类：{category_l1}/{category_l2}，品牌：{brand}，特点：{tags 用顿号连接}，{description}"`。
- 中文分词统一用 `jieba`。

## Review Focus

系统最可能被"真人输入"打穿、而各任务测试天然不会覆盖的五类情况（每个都对应到拥有该代码的任务里的测试）：

1. **空白/空 query**：`search("   ")` 不得崩溃，返回空结果并让生成层给出"没找到"而非报错。→ Task 6 / Task 7 测试。
2. **零召回（没有任何匹配）**：生成层必须回答"没找到"，**绝不编造商品**。→ Task 6 测试。
3. **约束提取返回非法 JSON / 乱码**：`extract_constraints` 回退为空约束（全量召回），流水线继续。→ Task 5 测试。
4. **生成层返回非法 JSON / 缺字段**：`parse_output` 回退为"用检索到的商品名拼一个确定性 summary"，不崩溃。→ Task 6 测试。
5. **同一商品出现在多路召回（重复 id）**：`rrf_fuse` 必须去重，不重复计分。→ Task 2 测试。

---

### Task 1: 项目脚手架 + 商品数据模型 + 样例数据

**Files:**
- Create: `requirements.txt`
- Create: `data/products.csv`
- Create: `src/__init__.py`
- Create: `src/data_model.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_data_model.py`

**Interfaces:**
- Produces:
  - `Product`（dataclass）：`id: str, name: str, category_l1: str, category_l2: str, brand: str, price: float, spec: str, tags: list[str], description: str`
  - `load_products(path: str | Path) -> list[Product]`
  - `REQUIRED_FIELDS: list[str]`（除 `tags`、`description` 外均为必填）
  - conftest fixture `sample_products() -> list[Product]`（跨任务复用，含 3 个类目 8~10 条手工 Product，tags 覆盖"无糖/低脂/无咖啡因/儿童"等）

- [ ] **Step 1: 写失败测试 `tests/test_data_model.py`**

`test_load_products_valid`：加载一个含 2 行的临时 CSV（列名与 Product 字段一致，`tags` 用 `|` 分隔），断言返回 2 个 `Product`，字段值正确、`price` 为 `float`、`tags` 为 `list[str]`。
`test_load_products_invalid_price`：某行 `price` 为 `"abc"`，断言抛 `ValueError` 且消息含该行标识。
`test_load_products_missing_field`：某行缺 `brand`，断言抛 `ValueError`。
`test_tags_empty_allowed`：某行 `tags` 为空字符串，断言 `tags == []`（`description` 同）。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_data_model.py -v`
Expected: FAIL（`Product` / `load_products` 未定义）

- [ ] **Step 3: 实现 `data_model.py`**

`Product` 用 `@dataclass`；`load_products` 用标准库 `csv.DictReader` 读取，逐行校验 `REQUIRED_FIELDS` 非空、`price` 可 `float()`（失败抛 `ValueError` 并附行号）；`tags`/`description` 允许空，`tags` 按 `|` split 且过滤空串。`src/__init__.py`、`tests/__init__.py` 留空。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_data_model.py -v`
Expected: PASS

- [ ] **Step 5: 写 `requirements.txt`、`data/products.csv`、`tests/conftest.py`**

`requirements.txt`：`sentence-transformers`、`chromadb`、`rank-bm25`、`jieba`、`openai`、`pytest`（各一行，不 pin 版本）。
`data/products.csv`：表头 = Product 字段名；写 20~30 条覆盖饮料/零食/乳品/日用品/速食五大类，`tags` 用 `|` 分隔且精标注。
`tests/conftest.py`：实现 `sample_products()` fixture（返回 8~10 条手写 Product，含不同类目与标签）。

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "feat: 商品数据模型与样例数据"
```

---

### Task 2: 检索纯函数：结构化过滤 + RRF 融合

**Files:**
- Create: `src/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `Product`、`sample_products`（Task 1）
- Produces:
  - `FilterConstraints`（dataclass）：`category_l1: str | None, category_l2: str | None, brand: str | None, price_min: float | None, price_max: float | None`
  - `filter_products(products: list[Product], constraints: FilterConstraints) -> list[Product]`
  - `rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[str]`

- [ ] **Step 1: 写失败测试**

`test_filter_by_category`：约束 `category_l1="饮料"`，断言结果全为饮料类。
`test_filter_by_price_range`：约束 `price_min=3.0, price_max=5.0`，断言结果价格均在区间内。
`test_filter_no_constraints_returns_all`：全空约束 → 返回全部（顺序不变）。
`test_rrf_fuse_ranking`：`rrf_fuse([["a","b","c"], ["b","a"]], k=60)` 断言首位为 `"a"` 或 `"b"` 中 RRF 分更高者（用公式手算断言精确名次）。
`test_rrf_fuse_dedupes`：`rrf_fuse([["a","b"], ["a","c"]])` 断言输出长度 3、无重复 id、`"a"` 只出现一次。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_retrieval.py -v`
Expected: FAIL（`filter_products` / `rrf_fuse` 未定义）

- [ ] **Step 3: 实现 `filter_products` 与 `rrf_fuse`**

`filter_products`：对每个非 None 约束做等值/区间过滤。`rrf_fuse` 算法固定如下：

```python
def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: (-scores[d], d))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/retrieval.py tests/test_retrieval.py
git commit -m "feat: 结构化过滤与 RRF 融合"
```

---

### Task 3: 文本构造 + 分词 + BM25 召回

**Files:**
- Create: `src/ingestion.py`
- Modify: `src/retrieval.py`（新增 `bm25_search`）
- Test: `tests/test_ingestion.py`、`tests/test_retrieval.py`（追加）

**Interfaces:**
- Consumes: `Product`（Task 1）
- Produces:
  - `build_searchable_text(p: Product) -> str`（严格按模板，tags 用 `、` 连接）
  - `tokenize(text: str) -> list[str]`（jieba `lcut`）
  - `bm25_search(query: str, index, k: int = 30) -> list[str]`（返回商品 id 按 BM25 分降序）

- [ ] **Step 1: 写失败测试**

`test_build_searchable_text`：给一个含全部字段的 `Product`，断言输出恰为 `"零度可乐 330ml，分类：饮料/碳酸饮料，品牌：可口可乐，特点：无糖、无咖啡因、低热量，适合想提神又不想摄入糖分的人群"`。
`test_tokenize_chinese`：`tokenize("无糖可乐")` 断言返回非空 token 列表且不含空串。
`test_bm25_search_ranking`：用 `sample_products()` 构造索引（含分词文本），query `"可乐"`，断言含"可乐"字样的商品排在不含者之前。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_ingestion.py tests/test_retrieval.py -v`
Expected: FAIL

- [ ] **Step 3: 实现文本构造、分词、BM25 索引与召回**

`ingestion.py`：`build_searchable_text` 按模板 f-string 拼接；`tokenize` 调 `jieba.lcut`。
`retrieval.py`：`bm25_search` 用 `rank_bm25.BM25Okapi(index.tokenized_texts)` 对 query 分词后打分，返回分数降序的 id 列表。为避免重复建索引，BM25 对象在 Task 4 的 `StoreIndex` 中缓存，本任务的测试用临时构造的 `BM25Okapi` 直接验证排序逻辑。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_ingestion.py tests/test_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/ingestion.py src/retrieval.py tests/
git commit -m "feat: 可检索文本、分词与 BM25 召回"
```

---

### Task 4: StoreIndex 组装 + 向量召回

**Files:**
- Modify: `src/ingestion.py`（新增 `StoreIndex`、`build_index`）
- Modify: `src/retrieval.py`（新增 `vector_search`）
- Test: `tests/test_ingestion.py`（追加）、`tests/test_retrieval.py`（追加）

**Interfaces:**
- Consumes: `Product`、`build_searchable_text`、`tokenize`、`bm25`（Task 3）
- Produces:
  - `StoreIndex`（dataclass）：`products: list[Product]`, `searchable_texts: list[str]`, `tokenized_texts: list[list[str]]`, `bm25`, `collection`, `embedder`
  - `build_index(products: list[Product], embedder) -> StoreIndex`
  - `vector_search(query: str, index: StoreIndex, k: int = 30) -> list[str]`
  - `Embedder` 契约：任何含 `.encode(texts: list[str]) -> list[list[float]]` 的对象（测试用 FakeEmbedder）

- [ ] **Step 1: 写失败测试**

`test_build_index_shapes`：用 FakeEmbedder（返回固定长度向量）对 `sample_products()` 建索引，断言 `len(index.searchable_texts) == len(index.products)`、`len(index.tokenized_texts) == len(index.products)`、`index.bm25` 非 None、`index.collection.count() == len(index.products)`。
`test_vector_search_ordering`：FakeEmbedder 让 query 向量与某商品向量点积最大，断言 `vector_search(query, index, k=3)` 首位为该商品 id。
FakeEmbedder 设计：`encode` 返回按输入文本 hash 生成的确定性单位向量。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_ingestion.py tests/test_retrieval.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `StoreIndex`、`build_index`、`vector_search`**

`build_index`：构建 `searchable_texts`、`tokenized_texts`、`BM25Okapi`；用 `embedder.encode(searchable_texts)` 得到向量，写入 `chromadb.Client()`（`PersistentClient` 或 ephemeral）的 collection，每个 doc 的 `id` 用 `Product.id`，metadata 存 `name/category_l1/category_l2/brand/price`，collection 距离度量设为 cosine。
`vector_search`：`embedder.encode([query])` 后调 `collection.query(query_embeddings=..., n_results=k)`，返回 `ids[0]`。embedder 通过 `index` 间接取得（`StoreIndex` 增加 `embedder` 字段）。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_ingestion.py tests/test_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/ingestion.py src/retrieval.py tests/
git commit -m "feat: StoreIndex 组装与向量召回"
```

---

### Task 5: 查询解析 + search 编排（filter-then-rank）

**Files:**
- Modify: `src/retrieval.py`（新增 `extract_constraints`、`search`）
- Test: `tests/test_retrieval.py`（追加）

**Interfaces:**
- Consumes: `StoreIndex`、`filter_products`、`vector_search`、`bm25_search`、`rrf_fuse`（Task 2~4）
- Produces:
  - `extract_constraints(query: str, client) -> FilterConstraints`
  - `search(query: str, index: StoreIndex, client, top_n: int = 5) -> list[Product]`

- [ ] **Step 1: 写失败测试**

`test_extract_constraints_valid`：FakeClient 返回 `{"category_l1":"饮料","price_max":5}`，断言 `extract_constraints` 得到对应 `FilterConstraints`。
`test_extract_constraints_malformed`：FakeClient 返回非 JSON 字符串，断言返回全空约束（不抛异常）。
`test_search_filter_then_rank`：`sample_products()` + FakeEmbedder 建索引，`search("饮料", index, client, top_n=2)` 断言返回 ≤2 条且全部 `category_l1=="饮料"`。
`test_search_empty_query`：`search("   ", ...)` 返回空列表（不崩溃）。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_retrieval.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `extract_constraints` 与 `search`**

`extract_constraints`：构造 prompt 要求 LLM 输出 JSON（`category_l1/category_l2/brand/price_min/price_max`），用 `json.loads` 解析；任何异常（网络/解析/结构不符）→ 返回全空 `FilterConstraints`。
`search`：空白 query → `[]`；否则 `constraints = extract_constraints(query, client)` → `pool = filter_products(index.products, constraints)` → 在 pool 内做向量与 BM25 召回（若 pool 为空则返回 `[]`）→ `rrf_fuse` 得 top-20 → 取前 `top_n` 个 id → 返回对应 `Product`。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/retrieval.py tests/test_retrieval.py
git commit -m "feat: 查询解析与 filter-then-rank 编排"
```

---

### Task 6: 生成层

**Files:**
- Create: `src/generation.py`
- Test: `tests/test_generation.py`

**Interfaces:**
- Consumes: `Product`（Task 1）
- Produces:
  - `GenerationResult`（dataclass）：`summary: str`, `products: list[dict]`
  - `build_messages(query: str, products: list[Product]) -> list[dict]`
  - `parse_output(text: str) -> GenerationResult`
  - `generate_summary(query: str, products: list[Product], client) -> GenerationResult`

- [ ] **Step 1: 写失败测试**

`test_build_messages_contains_system_and_products`：断言 messages 第一条是含"绝不能编造"字样的 system 消息，且 user 消息含每个商品的 name 与 query。
`test_parse_output_valid`：输入标准 JSON（含 `summary` 与 `products`），断言字段正确。
`test_parse_output_malformed`：输入非 JSON，断言返回 `summary == "未找到合适的商品"`（或空商品时同）、`products == []`，不抛异常。
`test_generate_summary_no_products`：`products=[]` 时，**不调用 client**，直接返回 `summary == "没有找到相关商品"`、`products == []`。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_generation.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `generation.py`**

`build_messages`：system 固定文案（含"只能基于给定商品作答，绝不能编造给定列表之外的商品或信息"）+ user（query + top-N 商品 JSON，温度 `0.1`）。
`parse_output`：`json.loads` 解析；失败或缺 `summary` 字段 → 返回 `GenerationResult(summary="未找到合适的商品", products=[])`。
`generate_summary`：空 products → 直接返回"没有找到相关商品"；否则调 `client.chat.completions.create(...)`（要求 JSON 输出），`parse_output` 解析。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_generation.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/generation.py tests/test_generation.py
git commit -m "feat: LLM 生成层"
```

---

### Task 7: 端到端入口 app.py + CLI

**Files:**
- Create: `src/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `build_index`、`search`、`generate_summary`（Task 4~6）
- Produces:
  - `build_embedder(model: str = "BAAI/bge-small-zh-v1.5") -> Embedder`（`SentenceTransformer` 实例）
  - `build_client() -> OpenAI`（从 `LLM_BASE_URL`/`LLM_API_KEY` 读取）
  - `answer(query: str, index: StoreIndex, client) -> dict`（`{"summary": str, "products": list[dict]}`）
  - CLI `main()`（`python -m src.app "query"`）

- [ ] **Step 1: 写失败测试**

`test_answer_end_to_end`：用 `sample_products()` + FakeEmbedder + FakeClient 建索引，`answer("无糖饮料", index, fake_client)` 断言返回 dict，含非空 `summary` 字符串与 `products` 列表，且每个 product 含 `id/name/price`。
`test_answer_empty_query`：`answer("   ", ...)` 返回 `{"summary": "没有找到相关商品", "products": []}`，不崩溃。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_app.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 `app.py`**

`answer` 串联 `search` → `generate_summary`，把 `GenerationResult` 转成 dict 返回。`build_embedder` 用 `SentenceTransformer(model)`。`build_client` 用 `OpenAI(base_url=..., api_key=...)`。`main()` 解析 `sys.argv[1]`，加载 `data/products.csv` → `build_index` → `answer` → `print(json.dumps(..., ensure_ascii=False, indent=2))`。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_app.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/app.py tests/test_app.py
git commit -m "feat: 端到端入口与 CLI"
```

---

### Task 8: 可选 rerank + 评估集（M3）

**Files:**
- Create: `src/rerank.py`
- Create: `eval/queries.json`
- Create: `eval/evaluate.py`
- Create: `README.md`
- Test: `tests/test_rerank.py`、`tests/test_evaluate.py`

**Interfaces:**
- Consumes: `Product`、`search`、`StoreIndex`（Task 2~5）
- Produces:
  - `rerank(query: str, candidates: list[Product], reranker) -> list[Product]`
  - `evaluate(index, client, queries: list[dict], k: int = 5) -> dict`（返回 `{"recall_at_k": float, "mrr": float}`）

- [ ] **Step 1: 写失败测试**

`test_rerank_returns_subset`：FakeReranker 给候选打分，`rerank` 返回按分降序的候选子集（长度 ≤ 输入）。
`test_evaluate_perfect_recall`：评测集 query 的期望 id 排在检索结果第 1 位时，`evaluate(...)` 返回 `recall_at_k == 1.0` 且 `mrr == 1.0`。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_rerank.py tests/test_evaluate.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 rerank、评估、README**

`rerank`：用 `CrossEncoder("BAAI/bge-reranker-base")` 对 `(query, searchable_text)` 打分，返回按分降序的候选。
`evaluate`：对每条 query 调 `search`，计算 `recall@k`（前 k 个结果命中期望 id 的比例）与 `MRR`（首个命中位置的倒数的平均）。`eval/queries.json` 写 20~30 条 `(query, expected_ids)`，覆盖精确/模糊/带价格约束三类。`README.md` 写架构说明与运行步骤。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_rerank.py tests/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/rerank.py eval/ tests/ README.md
git commit -m "feat: 可选 rerank 与评估集"
```

---

## 实施里程碑对照

- M1（最小链路）：Task 1 → 3 → 4 → 6 → 7（纯向量 + 生成跑通；Task 2 的纯函数同时就位）。
- M2（混合检索）：Task 2 → 5（结构化过滤 + BM25 + RRF 覆盖精确检索）。
- M3（精排与评估）：Task 8。

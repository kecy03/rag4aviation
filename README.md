# Aviation RAG — 航空垂直领域大模型 RAG 系统

面向航空教学的检索增强生成系统，支持飞机构造、飞行操作、法规规章三大知识域的智能问答。

## 架构

```
原始文档 → 多格式加载 → 语义分块 → 向量存储(Chroma)
用户问题 → 查询改写 → 领域路由 → 混合检索(BM25+向量) → 重排 → LLM生成 → 答案+溯源
```

## 项目结构

```
aviation-rag/
├── config/             # 配置（.env → Settings）
├── embeddings/         # DashScope text-embedding-v4
├── ingestion/          # 文档加载、OCR、语义分块、元数据增强
├── storage/            # Chroma 多collection管理、建库CLI
├── retrieval/          # 检索：BM25、向量、混合(RRF)、重排、查询改写、路由
├── generation/         # LLM(DeepSeek)、提示词模板、RAG生成器
├── conversation/       # 对话历史、多轮引擎
├── agent/              # 领域智能体（结构/飞行/法规）
├── evaluation/         # 测试用例、LLM裁判、检索/生成指标
├── data/               # 原始文档（按域分目录）
├── scripts/            # 构建脚本、交互demo
└── tests/              # 单元测试
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek 和 DashScope API Key
```

### 3. 准备数据

将文档放入对应目录：
- `data/aircraft_structure/` — 飞机构造资料
- `data/flight_ops/` — 飞行操作资料
- `data/regulations/` — 法规规章资料

### 4. 构建知识库

```bash
bash scripts/build_all.sh
# 或单独构建：
python -m storage.populate --collection aircraft_structure --source-dir data/aircraft_structure --reset
```

### 5. 交互问答

```bash
python scripts/demo.py
```

### 6. 运行评估

```bash
pytest evaluation/test_e2e.py -v
```

## 核心特性

- **多格式支持**：PDF、DOCX、HTML、图片OCR
- **语义分块**：按章节/段落边界切分，保持知识完整性
- **混合检索**：BM25关键词 + 向量语义，RRF融合排序
- **BGE重排**：BAAI/bge-reranker-v2-m3 交叉编码器精排
- **查询改写**：口语→规范术语，代词消解
- **多轮对话**：上下文感知，支持追问
- **多知识库**：飞机构造/飞行操作/法规规章独立管理
- **多维度评估**：Recall@K、MRR、Faithfulness、Relevance、Correctness

## 技术栈

| 组件 | 选型 |
|------|------|
| LLM | DeepSeek v4-flash |
| Embedding | 阿里云 DashScope text-embedding-v4 |
| 向量数据库 | Chroma |
| 重排序 | BAAI/bge-reranker-v2-m3 |
| 中文分词 | jieba |
| 评估 | LLM-as-Judge (1-5 Likert) |

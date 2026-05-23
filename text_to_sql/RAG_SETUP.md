# BGE-M3 RAG系统使用指南

## 概述
本系统集成了基于BGE-M3的RAG（检索增强生成）功能，通过向量检索相关上下文信息来增强Text-to-SQL的准确性。

## 功能特点

### 1. 智能检索
- 使用BGE-M3模型进行语义向量化
- FAISS向量数据库进行高效检索
- 支持相似度阈值过滤

### 2. 知识库管理
- 自动从Excel文件加载表结构信息
- 自动从Excel文件加载SQL知识库
- 支持文本分块和重叠处理

### 3. 上下文增强
- 根据用户查询检索相关上下文
- 将上下文信息融入LLM的prompt
- 提高SQL生成的准确性

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- `sentence-transformers==2.2.2` - BGE-M3模型
- `faiss-cpu==1.7.4` - 向量数据库
- `numpy==1.24.3` - 数值计算
- `scikit-learn==1.3.0` - 机器学习工具

## 使用方法

### 1. 初始化RAG系统

#### 初始化所有数据库
```bash
python init_rag.py init
```

#### 初始化指定数据库
```bash
python init_rag.py init box
python init_rag.py init huomiao
python init_rag.py init uk
```

### 2. 测试RAG搜索

```bash
python init_rag.py test box
```

### 3. 在代码中使用

```python
from rag_system import BGERAGSystem

# 创建RAG系统
rag = BGERAGSystem(cache_dir="./rag_cache_box")

# 搜索相关文档
results = rag.search("查询学生信息", top_k=5, threshold=0.5)

# 获取相关上下文
context = rag.get_relevant_context("本月学生数")
```

## 系统架构

### 1. 知识库构建
```
Excel文件 → 文本分块 → BGE-M3向量化 → FAISS索引
```

### 2. 查询流程
```
用户查询 → BGE-M3向量化 → FAISS检索 → 上下文增强 → LLM生成SQL
```

### 3. 文件结构
```
rag_cache_[数据库名]/
├── faiss_index.bin    # FAISS向量索引
└── metadata.pkl       # 文档元数据
```

## 配置参数

### BGERAGSystem参数
- `model_name`: BGE模型名称（默认："BAAI/bge-m3"）
- `cache_dir`: 缓存目录（默认："./rag_cache"）

### 搜索参数
- `top_k`: 返回结果数量（默认：5）
- `threshold`: 相似度阈值（默认：0.5）
- `chunk_size`: 文本分块大小（默认：512）
- `overlap`: 分块重叠大小（默认：50）

## 性能优化

### 1. 模型加载
- BGE-M3模型首次加载较慢，后续会缓存
- 建议在服务启动时预加载模型

### 2. 索引优化
- 使用FAISS的IndexFlatIP索引，适合小规模数据
- 大规模数据可考虑使用IndexIVFFlat等索引

### 3. 内存管理
- 向量数据存储在内存中，注意内存使用
- 可通过调整chunk_size控制内存占用

## 故障排除

### 1. 模型下载失败
```bash
# 手动下载模型
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

### 2. FAISS安装问题
```bash
# 安装CPU版本
pip install faiss-cpu

# 或安装GPU版本（需要CUDA）
pip install faiss-gpu
```

### 3. 内存不足
- 减少chunk_size
- 降低top_k值
- 增加相似度阈值

## 示例查询

### 表结构查询
```
查询: "学生表有哪些字段？"
检索: 表结构信息 → 字段列表
```

### SQL知识查询
```
查询: "如何统计用户数量？"
检索: 相关SQL示例 → COUNT查询
```

### 复杂查询
```
查询: "本月新增学生数"
检索: 学生表结构 + 时间相关SQL → 带时间条件的COUNT查询
```

## 监控和日志

### 1. 检索质量监控
- 记录检索结果的相关性分数
- 统计检索成功率
- 分析用户查询模式

### 2. 性能监控
- 检索响应时间
- 内存使用情况
- 索引大小变化

## 扩展功能

### 1. 多模态支持
- 支持图片、文档等多种数据类型
- 集成OCR、文档解析等功能

### 2. 实时更新
- 支持知识库的实时更新
- 增量索引构建

### 3. 个性化推荐
- 基于用户历史查询的个性化推荐
- 查询意图识别和分类

作者: Jamesenh 
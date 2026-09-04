# 向量检索与知识库同步

当前系统的检索链路不再使用早期 FAISS 缓存目录。表结构、SQL 知识库和业务名词的向量统一写入 Hologres/PostgreSQL 的 `knowledge` schema。

## 核心表

建表脚本：

```bash
psql "$DATABASE_URL" -f sql/create_knowledge_tables.sql
psql "$DATABASE_URL" -f sql/create_table_embeddings.sql
psql "$DATABASE_URL" -f sql/create_db_metadata.sql
```

相关表：

- `knowledge.table_embeddings`：表结构向量，包含 `local_embedding` 和 `doubao_embedding`
- `knowledge.db_knowledge`：SQL 示例知识库
- `knowledge.business_glossary`：业务名词
- `knowledge.db_metadata`：表结构 fingerprint，用于判断是否需要增量重建

## 向量模型

默认本地模型：

```bash
EMBEDDING_PROVIDER=local
SENTENCE_TRANSFORMER_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

可切到 Ark embedding API：

```bash
EMBEDDING_PROVIDER=api
ARK_EMBEDDING_API_KEY=your-key
ARK_EMBEDDING_MODEL=doubao-embedding-vision-251215
```

后台监控可同时补多列：

```bash
VECTOR_MONITOR_PROVIDERS=local,api
```

## 同步方式

应用启动后会调用 `start_vector_monitor()`，定期检查表结构变化并补齐缺失向量。也可以手动重建：

```bash
python schema_monitor.py --db-name hologres --rebuild
```

页面侧可通过知识库管理接口重建知识库和业务名词向量：

```bash
POST /knowledge/api/rebuild_vectors
```

表结构搜索由 `core.vector_search.TableSchemaSearcher` 执行：

- 优先使用 Hologres Proxima SQL 路径
- SQL 路径不可用时自动切到 numpy fallback
- 支持按 schema 过滤

## 上传匹配中的使用

上传匹配的业务目标表名只在进入模板配置或表搜索时读取当前 schema 下的 live metadata；选中具体目标表后才读取该表字段。只有在需要智能推荐目标表时才调用向量检索。

当前业务模板配置来自 `knowledge.upload_match_template`，不再从 `config.py` 静态配置、`default` 配置或上传表名兜底配置中合并。

## 故障排查

如果检索结果为空：

- 确认 `sql/create_knowledge_tables.sql` 已执行
- 检查 `knowledge.table_embeddings` 是否有对应 `db_name` 和向量列
- 本地模型首次加载较慢，确认 `sentence-transformers` 可正常下载/读取模型
- API 模型需要设置 `ARK_EMBEDDING_API_KEY` 或 `ARK_API_KEY`
- 查看日志中是否出现 “Hologres Proxima 路径不可用”，出现时系统会自动 fallback，不一定是故障

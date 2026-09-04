# Text2SQL 当前优化说明

## 上传匹配

当前上传匹配模板只走 `knowledge.upload_match_template` 的显式业务模板 key。

已清理的历史路径：

- 不再从 `config.py` 管理上传匹配模板
- 不再使用 `default` 模板作为全局默认配置
- 不再按上传表名查找或合并模板配置
- 前端不再把 `default`、`__draft__` 等历史/临时 key 展示为业务模板

性能相关优化：

- 目标表、模板字段选择从大 `<select>` 改为输入框 + 受限 datalist
- 字段候选按输入内容动态裁剪，避免一次性渲染全部字段
- 表名/字段查找增加前端缓存，切库或切 schema 时失效
- 页面初始化只加载模板、上传目标表名和历史，不预加载业务目标表字段
- 业务目标表名在进入模板配置或表搜索时通过 `/api/db_tables?include_columns=0` 轻量读取
- 业务目标表字段在选中具体表后通过 `/api/upload_table_columns` 单表读取
- 上传历史只先取最近表名，选中某张目标表后再读取该表字段

## 后端与数据库

当前应用元数据统一存放在 `knowledge` schema：

- `knowledge.app_user`
- `knowledge.db_knowledge`
- `knowledge.business_glossary`
- `knowledge.book_code_knowledge`
- `knowledge.table_embeddings`
- `knowledge.upload_match_template`
- `knowledge.query_log`

建表脚本：

- `sql/create_knowledge_tables.sql`
- `sql/create_table_embeddings.sql`
- `sql/create_db_metadata.sql`

仓库当前没有独立上传匹配迁移脚本或模板 seed 导入脚本；旧数据通过运行时过滤和建表脚本中的当前字段定义保持一致。

## 向量检索

当前表结构检索由 `core.vector_search.TableSchemaSearcher` 执行，向量存放在 `knowledge.table_embeddings`。

- 本地模型写入 `local_embedding`
- Ark embedding API 写入 `doubao_embedding`
- 优先使用 Hologres Proxima SQL 路径
- SQL 路径不可用时自动 fallback 到 numpy
- `schema_monitor.py` 和应用启动监控负责增量同步

早期 FAISS 缓存目录、`BGERAGSystem`、`init_rag.py` 这套说明已废弃。

## 可靠性

- `execute_nl_query` 支持请求取消和限流
- 查询日志统一写入 `knowledge.query_log`
- 登录用户、权限和同步状态写入 `knowledge.app_user`
- SQL 示例和业务名词支持按用户作用域管理

## 运行建议

初始化：

```bash
psql "$DATABASE_URL" -f sql/create_knowledge_tables.sql
psql "$DATABASE_URL" -f sql/create_table_embeddings.sql
psql "$DATABASE_URL" -f sql/create_db_metadata.sql
```

重建向量：

```bash
python schema_monitor.py --db-name hologres --rebuild
```

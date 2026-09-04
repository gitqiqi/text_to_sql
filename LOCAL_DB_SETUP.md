# Text2SQL 数据库设置

当前应用以 PostgreSQL/Hologres 作为主存储，应用元数据统一放在 `knowledge` schema。历史的本地 MySQL 日志库和 `query_logs` 表不再使用。

## 环境变量

在 `.env` 中配置主库连接：

```bash
DB_HOLOGRES_HOST=your-host
DB_HOLOGRES_PORT=80
DB_HOLOGRES_DATABASE=your-db
DB_HOLOGRES_USER=your-user
DB_HOLOGRES_PASSWORD=your-password
DB_HOLOGRES_SSLMODE=prefer
```

登录引导账号可选：

```bash
APP_ADMIN_PASSWORD=change-me
# 或 APP_ADMIN_PASSWORD_HASH=...
```

## 建表

初始化或补齐应用表：

```bash
psql "$DATABASE_URL" -f sql/create_knowledge_tables.sql
psql "$DATABASE_URL" -f sql/create_table_embeddings.sql
psql "$DATABASE_URL" -f sql/create_db_metadata.sql
```

主要表：

- `knowledge.app_user`：登录用户与权限资料
- `knowledge.db_knowledge`：SQL 知识库
- `knowledge.business_glossary`：业务名词
- `knowledge.book_code_knowledge`：代码知识索引
- `knowledge.upload_match_template`：上传匹配业务模板
- `knowledge.query_log`：自然语言查询和上传匹配日志
- `knowledge.table_embeddings`：表结构向量
- `knowledge.db_metadata`：表结构 fingerprint

## 上传匹配模板

上传匹配模板存放在 `knowledge.upload_match_template`，当前运行时只读取显式业务模板 key。

仓库当前没有独立的上传匹配迁移脚本或 seed 导入脚本；历史 `default`、`__draft__` 和按上传表名兜底的配置不会再被前端展示或后端合并使用。

## 账号同步

如需从业务源表导入用户资料，可执行：

```bash
psql "$DATABASE_URL" -f sql/sync_app_user_from_admin_info.sql
```

默认源表为 `bi.dim_org_admin_user_info_hf`。如果源字段不同，先按实际库表调整 SQL。

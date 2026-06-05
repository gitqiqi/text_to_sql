-- 表结构指纹元数据表（schema 变化检测用）
-- schema: knowledge

CREATE TABLE IF NOT EXISTS knowledge.db_metadata (
    db_name VARCHAR(50) PRIMARY KEY,
    schema_fingerprint VARCHAR(64) NOT NULL,
    last_updated TIMESTAMP DEFAULT NOW()
);

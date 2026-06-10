-- SQL 知识库 & 业务名词表（需在目标数据库中提前创建）
-- schema: knowledge

CREATE TABLE IF NOT EXISTS knowledge.db_knowledge (
    id SERIAL PRIMARY KEY,
    db_name VARCHAR(50) NOT NULL,
    question TEXT NOT NULL,
    sql TEXT NOT NULL,
    local_embedding REAL[] CHECK(array_ndims(local_embedding) = 1 AND array_length(local_embedding, 1) = 384),
    doubao_embedding REAL[] CHECK(array_ndims(doubao_embedding) = 1 AND array_length(doubao_embedding, 1) = 2048),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge.business_glossary (
    id SERIAL PRIMARY KEY,
    db_name VARCHAR(50) NOT NULL,
    term VARCHAR(200) NOT NULL,
    definition TEXT NOT NULL,
    local_embedding REAL[] CHECK(array_ndims(local_embedding) = 1 AND array_length(local_embedding, 1) = 384),
    doubao_embedding REAL[] CHECK(array_ndims(doubao_embedding) = 1 AND array_length(doubao_embedding, 1) = 2048),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

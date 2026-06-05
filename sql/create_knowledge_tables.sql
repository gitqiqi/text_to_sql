-- SQL 知识库 & 业务名词表（需在目标数据库中提前创建）
-- schema: knowledge

CREATE TABLE IF NOT EXISTS knowledge.db_knowledge (
    id integer NOT NULL default nextval('hg_recyclebin.db_knowledge_id_seq_609856f7262d4c0d9df663f8e551cbbc'::regclass),
    db_name character varying(50) NOT NULL,
    question text NOT NULL,
    sql text NOT NULL,
    local_embedding real[],
    doubao_embedding real[],
    created_at timestamp without time zone default now(),
    updated_at timestamp without time zone default now()
    ,PRIMARY KEY (id)
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

-- Text2SQL 应用表
-- schema: knowledge

CREATE SCHEMA IF NOT EXISTS knowledge;

-- 应用用户表：从 bi.dim_org_admin_user_info_hf 抽取必要账号信息后落到这里。
-- 登录使用 password_hash；密码通常随源平台同步，admin 引导账号仍可由 APP_ADMIN_PASSWORD / APP_ADMIN_PASSWORD_HASH 初始化。
CREATE TABLE IF NOT EXISTS knowledge.app_user (
    admin_id BIGINT PRIMARY KEY,
    user_name TEXT,
    mobile TEXT,
    we_user_id TEXT,
    employee_id TEXT,
    role_id BIGINT,
    role_name TEXT,
    admin_organ_id BIGINT,
    organ_name TEXT,
    teacher_uid BIGINT,
    subject BIGINT,
    status BIGINT DEFAULT 1,
    is_full_view BIGINT DEFAULT 0,
    permission_type BIGINT,
    permission_scope BIGINT,
    admin_organ_ids TEXT,
    parent_ids TEXT,
    password_hash TEXT,
    last_login_at TIMESTAMP,
    source_create_date TIMESTAMP WITH TIME ZONE,
    source_update_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_app_user_mobile ON knowledge.app_user (mobile);
CREATE INDEX IF NOT EXISTS idx_app_user_user_name ON knowledge.app_user (user_name);
CREATE INDEX IF NOT EXISTS idx_app_user_employee_id ON knowledge.app_user (employee_id);

-- SQL 知识库
CREATE TABLE IF NOT EXISTS knowledge.db_knowledge (
    id SERIAL PRIMARY KEY,
    db_name VARCHAR(50) NOT NULL,
    question TEXT NOT NULL,
    sql TEXT NOT NULL,
    local_embedding REAL[] CHECK(array_ndims(local_embedding) = 1 AND array_length(local_embedding, 1) = 384),
    doubao_embedding REAL[] CHECK(array_ndims(doubao_embedding) = 1 AND array_length(doubao_embedding, 1) = 2048),
    created_by_admin_id BIGINT,
    created_by_user_name TEXT,
    updated_by_admin_id BIGINT,
    updated_by_user_name TEXT,
    visibility_scope TEXT DEFAULT 'self',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE knowledge.db_knowledge ADD COLUMN IF NOT EXISTS local_embedding REAL[];
ALTER TABLE knowledge.db_knowledge ADD COLUMN IF NOT EXISTS doubao_embedding REAL[];
ALTER TABLE knowledge.db_knowledge ADD COLUMN IF NOT EXISTS created_by_admin_id BIGINT;
ALTER TABLE knowledge.db_knowledge ADD COLUMN IF NOT EXISTS created_by_user_name TEXT;
ALTER TABLE knowledge.db_knowledge ADD COLUMN IF NOT EXISTS updated_by_admin_id BIGINT;
ALTER TABLE knowledge.db_knowledge ADD COLUMN IF NOT EXISTS updated_by_user_name TEXT;
ALTER TABLE knowledge.db_knowledge ADD COLUMN IF NOT EXISTS visibility_scope TEXT DEFAULT 'self';

CREATE INDEX IF NOT EXISTS idx_db_knowledge_owner ON knowledge.db_knowledge (db_name, created_by_admin_id);

-- 业务名词
CREATE TABLE IF NOT EXISTS knowledge.business_glossary (
    id SERIAL PRIMARY KEY,
    db_name VARCHAR(50) NOT NULL,
    term VARCHAR(200) NOT NULL,
    definition TEXT NOT NULL,
    local_embedding REAL[] CHECK(array_ndims(local_embedding) = 1 AND array_length(local_embedding, 1) = 384),
    doubao_embedding REAL[] CHECK(array_ndims(doubao_embedding) = 1 AND array_length(doubao_embedding, 1) = 2048),
    created_by_admin_id BIGINT,
    created_by_user_name TEXT,
    updated_by_admin_id BIGINT,
    updated_by_user_name TEXT,
    visibility_scope TEXT DEFAULT 'self',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE knowledge.business_glossary ADD COLUMN IF NOT EXISTS local_embedding REAL[];
ALTER TABLE knowledge.business_glossary ADD COLUMN IF NOT EXISTS doubao_embedding REAL[];
ALTER TABLE knowledge.business_glossary ADD COLUMN IF NOT EXISTS created_by_admin_id BIGINT;
ALTER TABLE knowledge.business_glossary ADD COLUMN IF NOT EXISTS created_by_user_name TEXT;
ALTER TABLE knowledge.business_glossary ADD COLUMN IF NOT EXISTS updated_by_admin_id BIGINT;
ALTER TABLE knowledge.business_glossary ADD COLUMN IF NOT EXISTS updated_by_user_name TEXT;
ALTER TABLE knowledge.business_glossary ADD COLUMN IF NOT EXISTS visibility_scope TEXT DEFAULT 'self';

CREATE INDEX IF NOT EXISTS idx_business_glossary_owner ON knowledge.business_glossary (db_name, created_by_admin_id);

-- 仓库代码知识：函数 / 类 / 文件级上下文索引。
CREATE TABLE IF NOT EXISTS knowledge.book_code_knowledge (
    id BIGSERIAL PRIMARY KEY,
    db_name VARCHAR(50) NOT NULL,
    repo_name TEXT NOT NULL,
    repo_path TEXT,
    branch_name TEXT,
    commit_hash TEXT,
    file_path TEXT NOT NULL,
    file_name TEXT,
    module_path TEXT,
    language TEXT,
    symbol_type TEXT NOT NULL,
    symbol_name TEXT NOT NULL DEFAULT '',
    qualified_name TEXT NOT NULL DEFAULT '',
    parent_name TEXT,
    signature TEXT,
    docstring TEXT,
    leading_comments TEXT,
    code_text TEXT,
    context_text TEXT,
    imports_json JSONB,
    references_json JSONB,
    context_json JSONB,
    line_start INTEGER,
    line_end INTEGER,
    symbol_order INTEGER,
    vector_text TEXT,
    local_embedding REAL[] CHECK(array_ndims(local_embedding) = 1 AND array_length(local_embedding, 1) = 384),
    doubao_embedding REAL[] CHECK(array_ndims(doubao_embedding) = 1 AND array_length(doubao_embedding, 1) = 2048),
    text_hash VARCHAR(64),
    source_hash VARCHAR(64),
    is_active BOOLEAN DEFAULT TRUE,
    last_seen_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS repo_name TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS repo_path TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS branch_name TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS commit_hash TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS file_path TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS file_name TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS module_path TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS language TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS symbol_type TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS symbol_name TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS qualified_name TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS parent_name TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS signature TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS docstring TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS leading_comments TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS code_text TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS context_text TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS imports_json JSONB;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS references_json JSONB;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS context_json JSONB;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS line_start INTEGER;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS line_end INTEGER;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS symbol_order INTEGER;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS vector_text TEXT;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS local_embedding REAL[];
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS doubao_embedding REAL[];
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS text_hash VARCHAR(64);
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64);
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMP;
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();
ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();

CREATE UNIQUE INDEX IF NOT EXISTS idx_book_code_knowledge_key
    ON knowledge.book_code_knowledge (db_name, repo_name, file_path, symbol_type, qualified_name);
CREATE INDEX IF NOT EXISTS idx_book_code_knowledge_active
    ON knowledge.book_code_knowledge (db_name, repo_name, is_active, updated_at);
CREATE INDEX IF NOT EXISTS idx_book_code_knowledge_file
    ON knowledge.book_code_knowledge (db_name, repo_name, file_path, symbol_order);
CREATE INDEX IF NOT EXISTS idx_book_code_knowledge_symbol
    ON knowledge.book_code_knowledge (db_name, repo_name, symbol_name);
CREATE INDEX IF NOT EXISTS idx_book_code_knowledge_commit
    ON knowledge.book_code_knowledge (db_name, repo_name, commit_hash);

-- 上传匹配模板：页面“模板配置 / 模板列表”的持久化镜像。
CREATE TABLE IF NOT EXISTS knowledge.upload_match_template (
    db_name VARCHAR(50) NOT NULL,
    template_key VARCHAR(200) NOT NULL,
    label TEXT,
    description TEXT,
    keyword_column TEXT,
    match_table TEXT,
    match_field TEXT,
    match_field_display TEXT,
    match_mode TEXT,
    target_filter TEXT,
    sql_text TEXT,
    return_fields_json TEXT,
    config_json TEXT,
    status TEXT DEFAULT 'active',
    created_by TEXT,
    updated_by TEXT,
    created_by_admin_id BIGINT,
    created_by_user_name TEXT,
    updated_by_admin_id BIGINT,
    updated_by_user_name TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (db_name, template_key)
);

ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS return_fields_json TEXT;
ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS config_json TEXT;
ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active';
ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS created_by TEXT;
ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS updated_by TEXT;
ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS created_by_admin_id BIGINT;
ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS created_by_user_name TEXT;
ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS updated_by_admin_id BIGINT;
ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS updated_by_user_name TEXT;
ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();
ALTER TABLE knowledge.upload_match_template ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();
-- Hologres 中如果客户端开启显式事务，请在 ALTER 提交后单独执行下面的 UPDATE。
UPDATE knowledge.upload_match_template SET status = 'active' WHERE status IS NULL;

-- 查询运行日志
CREATE TABLE IF NOT EXISTS knowledge.query_log (
    id BIGSERIAL PRIMARY KEY,
    db_name TEXT,
    nl_query TEXT,
    schema_filter TEXT,
    search_mode TEXT,
    selected_table TEXT,
    top_k INTEGER,
    matched_tables TEXT,
    generated_sql TEXT,
    execute_status TEXT,
    error_message TEXT,
    result_rows INTEGER,
    search_duration_ms DOUBLE PRECISION,
    llm_duration_ms DOUBLE PRECISION,
    sql_exec_duration_ms DOUBLE PRECISION,
    total_duration_ms DOUBLE PRECISION,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    llm_calls INTEGER,
    admin_id BIGINT,
    request_id TEXT,
    session_id TEXT,
    client_ip TEXT,
    user_agent TEXT,
    action_type TEXT,
    visibility_scope TEXT DEFAULT 'self',
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE knowledge.query_log ADD COLUMN IF NOT EXISTS admin_id BIGINT;
ALTER TABLE knowledge.query_log ADD COLUMN IF NOT EXISTS request_id TEXT;
ALTER TABLE knowledge.query_log ADD COLUMN IF NOT EXISTS session_id TEXT;
ALTER TABLE knowledge.query_log ADD COLUMN IF NOT EXISTS client_ip TEXT;
ALTER TABLE knowledge.query_log ADD COLUMN IF NOT EXISTS user_agent TEXT;
ALTER TABLE knowledge.query_log ADD COLUMN IF NOT EXISTS action_type TEXT;
ALTER TABLE knowledge.query_log ADD COLUMN IF NOT EXISTS visibility_scope TEXT DEFAULT 'self';

CREATE INDEX IF NOT EXISTS idx_query_log_owner ON knowledge.query_log (db_name, admin_id, created_at);
CREATE INDEX IF NOT EXISTS idx_query_log_request_id ON knowledge.query_log (request_id);

-- DataWorks 节点代码知识表初始化。
-- 用有 DDL 权限的管理员账号执行一次，然后执行 grant_knowledge_permissions.sql
-- 给调度运行账号授权。

CREATE SCHEMA IF NOT EXISTS knowledge;

CREATE TABLE IF NOT EXISTS knowledge.dataworks_node_knowledge (
    id BIGSERIAL PRIMARY KEY,
    db_name VARCHAR(50) NOT NULL,
    project_id BIGINT NOT NULL,
    project_identifier TEXT,
    workspace_region TEXT,
    node_key TEXT NOT NULL,
    node_id TEXT,
    file_id TEXT,
    node_name TEXT,
    file_name TEXT,
    file_folder_path TEXT,
    absolute_folder_path TEXT,
    file_type TEXT,
    use_type TEXT,
    connection_name TEXT,
    owner TEXT,
    last_edit_user TEXT,
    commit_status TEXT,
    auto_parsing BOOLEAN,
    is_maxcompute BOOLEAN,
    current_version TEXT,
    file_description TEXT,
    source_modified_at TIMESTAMP,
    content TEXT,
    input_list JSONB,
    output_list JSONB,
    dependent_node_ids JSONB,
    upstream_nodes JSONB,
    upstream_tables JSONB,
    output_tables JSONB,
    node_configuration JSONB,
    file_payload JSONB,
    text_hash VARCHAR(64),
    source_hash VARCHAR(64),
    last_seen_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
) WITH (
    orientation = 'column',
    storage_format = 'orc',
    bitmap_columns = 'db_name,project_id,node_key,node_id,file_id,file_type,use_type,commit_status,is_active',
    dictionary_encoding_columns = 'db_name:auto,project_id:auto,project_identifier:auto,workspace_region:auto,node_key:auto,node_id:auto,file_id:auto,node_name:auto,file_folder_path:auto,absolute_folder_path:auto,file_type:auto,use_type:auto,connection_name:auto,owner:auto,last_edit_user:auto,commit_status:auto,current_version:auto,file_description:auto,text_hash:auto,source_hash:auto',
    distribution_key = 'id',
    table_group = 'db_tg_default',
    table_storage_mode = 'any',
    time_to_live_in_seconds = '3153600000'
);

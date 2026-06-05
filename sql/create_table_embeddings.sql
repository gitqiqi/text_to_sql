CREATE TABLE knowledge.table_embeddings (
    id BIGSERIAL PRIMARY KEY,
    db_name VARCHAR(50) NOT NULL,
    schema_name VARCHAR(100),
    table_name VARCHAR(100) NOT NULL,
    table_comment TEXT,
    column_info JSONB,
    vector_text TEXT,
    local_embedding REAL[] CHECK (array_ndims(local_embedding) = 1 AND array_length(local_embedding, 1) = 384),
    doubao_embedding REAL[] CHECK (array_ndims(doubao_embedding) = 1 AND array_length(doubao_embedding, 1) = 2048),
    text_hash VARCHAR(64),
    schema_hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
) WITH (
    orientation = 'column',
    storage_format = 'orc',
    bitmap_columns = 'db_name,schema_name,table_name',
    dictionary_encoding_columns = 'db_name:auto,schema_name:auto,table_name:auto,table_comment:auto,vector_text:auto,text_hash:auto,schema_hash:auto',
    distribution_key = 'id',
    proxima_vectors = '{
        "local_embedding": {
            "algorithm": "Graph",
            "distance_method": "SquaredEuclidean",
            "builder_params": {
                "min_flush_proxima_row_count": 1000,
                "min_compaction_proxima_row_count": 1000,
                "max_total_size_to_merge_mb": 2000
            }
        },
        "doubao_embedding": {
            "algorithm": "Graph",
            "distance_method": "SquaredEuclidean",
            "builder_params": {
                "min_flush_proxima_row_count": 1000,
                "min_compaction_proxima_row_count": 1000,
                "max_total_size_to_merge_mb": 2000
            }
        }
    }',
    table_group = 'db_tg_default',
    table_storage_mode = 'any',
    time_to_live_in_seconds = '3153600000'
);

-- bitmap_columns 已在 WITH 中配置，无需额外建索引

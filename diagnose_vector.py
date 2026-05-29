"""诊断向量检索：为什么某张表没进 Top 10？"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sentence_transformers import SentenceTransformer

from core.utils import SENTENCE_TRANSFORMER_MODEL
from core.knowledge import KnowledgeBase

# ============= 改这里 =============
DB_NAME = 'hologres'
QUERY = '年级维度26春在读学生数以及续报id为92，93续报人数'
SCHEMA_FILTER = 'bi'  # 跟你前端选的一致；想看全库改成 None
TARGET_TABLE = 'dim_org_class_member_hf'  # 你期望出现但没进 Top 10 的表
# ==================================

print(f"\n{'='*70}")
print(f"查询: {QUERY}")
print(f"目标表: {TARGET_TABLE}")
print(f"{'='*70}\n")

kb = KnowledgeBase(DB_NAME)
records, embeddings = kb._load_all_vectors_cached()
print(f"总向量数: {len(records)}")

if SCHEMA_FILTER:
    schemas = set(s.strip().lower() for s in SCHEMA_FILTER.split(','))
    keep = [i for i, r in enumerate(records) if (r.get('schema') or '').lower() in schemas]
    records = [records[i] for i in keep]
    embeddings = embeddings[keep]
    print(f"按 schema={SCHEMA_FILTER} 过滤后: {len(records)} 张表")

# 编码 query
model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
query_emb = model.encode([QUERY], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)[0]

# 算所有距离
diff = embeddings - query_emb.astype(np.float32)
distances = np.einsum('ij,ij->i', diff, diff)

# 排序
sorted_idx = np.argsort(distances)

# 找目标表的排名
target_rank = None
target_distance = None
target_vector_text = None
for rank, idx in enumerate(sorted_idx, 1):
    if records[idx]['table_name'] == TARGET_TABLE:
        target_rank = rank
        target_distance = float(distances[idx])
        target_vector_text = records[idx]['vector_text']
        break

print(f"\n{'─'*70}")
print(f"📊 Top 15（在 schema={SCHEMA_FILTER or '全部'} 范围内）")
print(f"{'─'*70}")
for rank, idx in enumerate(sorted_idx[:15], 1):
    r = records[idx]
    full = f"{r.get('schema','')}.{r['table_name']}"
    marker = " ⭐ 目标表" if r['table_name'] == TARGET_TABLE else ""
    print(f"  {rank:2d}. dist={float(distances[idx]):.4f}  {full}{marker}")

print(f"\n{'─'*70}")
print(f"🎯 目标表 {TARGET_TABLE} 的情况")
print(f"{'─'*70}")
if target_rank is None:
    print(f"❌ 在当前 schema 过滤下完全找不到这张表！")
    print(f"   可能：1) 这张表没建过向量；2) 它的 schema 不是 '{SCHEMA_FILTER}'")
else:
    print(f"  排名: 第 {target_rank} 名（共 {len(records)} 张表）")
    print(f"  距离: {target_distance:.4f}")
    print(f"  与 Top 1 的差距: +{target_distance - float(distances[sorted_idx[0]]):.4f}")
    print(f"\n  vector_text（被向量化的文本）:")
    print(f"  {'─'*60}")
    # 截断显示前 800 字符
    show_text = target_vector_text[:800]
    if len(target_vector_text) > 800:
        show_text += f"\n  ...(后面还有 {len(target_vector_text)-800} 字符)"
    for line in show_text.split('\n'):
        print(f"  {line}")
    print(f"  {'─'*60}")

# 显示 Top 1 的 vector_text 做对比
print(f"\n{'─'*70}")
print(f"🥇 Top 1 表的 vector_text（用来对比）")
print(f"{'─'*70}")
top1 = records[sorted_idx[0]]
print(f"  表名: {top1.get('schema','')}.{top1['table_name']}")
print(f"  距离: {float(distances[sorted_idx[0]]):.4f}")
top1_text = top1['vector_text'][:800]
if len(top1['vector_text']) > 800:
    top1_text += f"\n  ...(后面还有 {len(top1['vector_text'])-800} 字符)"
for line in top1_text.split('\n'):
    print(f"  {line}")

print()

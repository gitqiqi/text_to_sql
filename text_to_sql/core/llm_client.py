# core/llm_client.py - 豆包 LLM 客户端 + 全部 prompt
import os
import re
from typing import Any, Dict, List, Optional

from volcenginesdkarkruntime import Ark

from .utils import (
    MAX_TABLE_LENGTH_PER_BATCH,
    MAX_BATCHES,
    MIN_TABLES_PER_BATCH,
    monitor_function,
    retry,
    extract_final_sql,
)


class DouBaoClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = os.getenv('ARK_MODEL')
        if not self.model:
            raise ValueError('ARK_MODEL environment variable is required')
        self.client = Ark(api_key=self.api_key)

    @monitor_function
    def generate_text(self, nl_query: str, formatted_tables: str, knowledge_json: Any, glossary: Optional[List[Dict]] = None) -> str:
        """生成 SQL，支持拆分长表结构进行多轮对话"""

        formatted_knowledge = self._format_knowledge(knowledge_json)
        formatted_glossary = self._format_glossary(glossary or [])

        print(f"    ├─ 表结构信息总长度: {len(formatted_tables)} 字符")
        print(f"    ├─ 知识库示例长度: {len(formatted_knowledge)} 字符")
        print(f"    ├─ 业务名词数量: {len(glossary or [])}")

        if len(formatted_tables) <= MAX_TABLE_LENGTH_PER_BATCH:
            print(f"    ├─ 表结构未超长，直接处理")
            return self._generate_single_request(nl_query, formatted_tables, formatted_knowledge, formatted_glossary)

        print(f"    ├─ 表结构超长，开始拆分（每批最大 {MAX_TABLE_LENGTH_PER_BATCH} 字符）")
        table_batches = self._split_tables(formatted_tables)

        if len(table_batches) > MAX_BATCHES:
            print(f"    ⚠️ 表批次过多({len(table_batches)}批)，合并到{MAX_BATCHES}批")
            table_batches = self._merge_batches(table_batches, MAX_BATCHES)

        print(f"    ├─ 拆分为 {len(table_batches)} 批处理")

        all_sql_candidates = []

        for i, batch_tables in enumerate(table_batches, 1):
            print(f"    ├─ 处理第 {i}/{len(table_batches)} 批（{len(batch_tables)} 字符）")

            batch_prompt = f"""你是一个专业的SQL查询生成器。
这是第 {i}/{len(table_batches)} 批表结构。需要先思考再生成SQL。

## 输出格式：
## 思考过程
1. 这批表中是否有能回答问题的表？如果没有，下面直接写 "-- 本批无相关表"
2. 如果有，列出涉及的表和字段（逐个核对字段在表结构中存在，**留意每个字段的数据类型**）
3. 写出 JOIN 和 WHERE 条件依据，注意类型不匹配时加 cast

## 最终SQL
<只放可执行的SQL，或 "-- 本批无相关表">

## 重要规则：
- 严格只使用下方列出的表名和字段名，禁止编造字段
- SQL必须以SELECT或WITH开头

## 字段格式说明：
每个字段格式为 `字段名 [数据类型] (注释)`，**字段名仅指方括号前的那段**。
引用时直接用字段名（如 `grade`），或 `表名.字段名`（如 `db_course_student.grade`）。
禁止把表名拼到字段名前，例如 `db_course_student_grade` 是错误的。

## ⚠️ Hologres 类型严格性：
Hologres 类型不一致会直接报错，必须主动加显式转换：
- 字段是 text **含时分秒**（如 `'2025-01-01 12:34:56'`）→ 用 `::timestamp`，禁止用 `::date`（会丢精度且可能报错）
- 字段是 text **只有日期**（如 `'2025-01-01'`）→ 用 `::date`
- 字段是 text 但要按数字比较：`field::bigint` 或 `field::numeric`
- JOIN 两侧类型不一致：给一边加 cast 比如 `a.id::bigint = b.uid`
- union 两表类型不一致：给一边加 cast 比如 `select id::bigint union all select uid::bigint`
- 数值字段比较禁止用引号：`WHERE id = 123` 而不是 `WHERE id = '123'`
- 留意字段注释里的格式说明（如 "格式: yyyy-mm-dd hh:mm:ss"）来判断转换方式

## 表结构（第{i}批）：
{batch_tables}

## 业务名词解释（必读，理解业务术语后再生成 SQL）：
{formatted_glossary}

## 知识库示例：
{formatted_knowledge}"""

            try:
                content = self._call_llm(
                    system_prompt=batch_prompt,
                    user_message=f"问题：{nl_query}\n\nSQL："
                )

                if content and not content.startswith("-- 本批无相关表") and "本批无相关表" not in content[:50]:
                    cleaned_sql = extract_final_sql(content)
                    if cleaned_sql and cleaned_sql.upper().startswith(('SELECT', 'WITH')):
                        all_sql_candidates.append({
                            'batch': i,
                            'sql': cleaned_sql,
                            'content': content
                        })
                        print(f"    │   └─ 第{i}批找到候选SQL")

            except Exception as e:
                print(f"    │   └─ 第{i}批请求失败: {e}")
                continue

        if all_sql_candidates:
            best_sql = self._select_best_sql(all_sql_candidates, nl_query)
            print(f"    ├─ 从 {len(all_sql_candidates)} 个候选中选择最佳SQL")
            return best_sql

        print(f"    ⚠️ 所有批次均未找到有效SQL")
        return ""

    def _generate_single_request(self, nl_query: str, formatted_tables: str, formatted_knowledge: str, formatted_glossary: str = "") -> str:
        """单次请求生成SQL"""

        if len(formatted_tables) > MAX_TABLE_LENGTH_PER_BATCH:
            print(f"    ⚠️ 单批表结构过长({len(formatted_tables)}字符)，截断到{MAX_TABLE_LENGTH_PER_BATCH}字符")
            formatted_tables = formatted_tables[:MAX_TABLE_LENGTH_PER_BATCH] + "\n...(表结构已截断)"

        content = self._call_llm(
            system_prompt=f"""你是一个专业的SQL查询生成器。你需要先一步步思考，再生成最终的SQL语句。

## 输出格式（必须严格遵守）：
你的输出必须分为两部分，使用以下标记分隔：

## 思考过程
1. 用户想问什么：用一句话改写问题，并列出用户期望的"输出指标"（比如：是否续报、是否囤课、续报状态…）
2. 涉及哪些表：从下方表结构中挑选，写出完整表名（schema.table）
3. 涉及哪些字段：列出每个字段的完整名称（表名.字段名）和**数据类型**，并**逐个核对**这些字段确实出现在下方"表结构"中。**特别留意类型**：如果字段类型与用法不匹配（比如要做日期比较但字段是 text，或要做数值运算但字段是 varchar），需要在 SQL 里加显式转换
4. **指标推导清单**（关键步骤，每个用户要的指标都必须写一行）：
   - 对每个"输出指标"，明确"用哪个真实字段算出来"
   - 如果**直接有同名/同义字段** → 直接用，写出字段名
   - 如果**没有现成字段，但能用其他真实字段计算/CASE WHEN 推导** → 写出推导逻辑（例：是否续报 = `CASE WHEN renewal_status = 1 THEN '是' ELSE '否' END`）
   - 如果**所有真实字段都无法推导出该指标** → 在思考过程里**明确标注 "无法推导：xxx 指标"**，最终SQL 里**直接省略该列**（绝不能用假设字段占位）
   - **绝对禁止**：写出注释如"假设字段"、"这里假设有 xxx 字段"、"实际需要根据真实表结构调整"——出现这些就是失败
5. 表之间如何关联（如有多表）：写出 JOIN 条件，**核对两侧字段类型一致**，不一致就加 cast
6. 过滤/分组/排序条件：写出 WHERE / GROUP BY / ORDER BY 的依据，注意类型转换

## 最终SQL
<这里只放可执行的SQL，以SELECT或WITH开头，不要任何markdown标记、注释、解释、不要 ``` 收尾、不要在SQL后面写"需要注意的是…"之类的说明文字>

## 重要规则：
1. **严格约束**：只能使用下方"表结构"中明确列出的表名和字段名，禁止使用任何未列出的表或字段，禁止猜测或编造字段名
2. **指标推导规则**：用户要的"指标"如果不是表里直接的字段，**必须用现有真实字段做计算/CASE WHEN/聚合**推导出来，禁止使用任何"假设字段"。如果实在推导不出来，宁可在结果里**省略该列**，也不要编造字段
3. 如果整个查询找不到相关的表或字段，"## 最终SQL" 后面只输出: -- 无相关表
4. 优先使用schema为bi里面的表
5. SQL必须以SELECT或WITH开头，不要使用```sql```标记，结尾也不要 ```
6. **SQL输出后立即结束**，不要追加任何中文说明、"需要注意的是"、"假设字段"之类的话

## 字段格式说明（非常重要！）：
表结构中每个字段的格式为：`字段名 [数据类型] (注释)`
- **字段名就是空格前那一段**，方括号 `[]` 和小括号 `()` 内的内容都不是字段名的一部分
- 在 SQL 中引用字段时，**直接使用字段名本身**，禁止把表名拼接到字段名前
- 如需限定字段所属的表，正确写法是 `表名.字段名`（用点号分隔），绝不要写成 `表名_字段名` 或 `db_表名_字段名`

## ⚠️ Hologres 类型严格性（非常重要！）：
Hologres 对类型匹配非常严格，类型不一致会直接报错。生成 SQL 时**必须主动加类型转换**：

1. **字符串/数字比较**：如果字段是 `text/varchar` 但要跟数字比较，加 `::int` 或 `::bigint`；反之字段是数值但要当文本，加 `::text`
   - 例：`WHERE id::bigint = 123`，`WHERE id::text LIKE '1%'`
2. **日期/时间字段比较**：要使用to_char(field, 'YYYY-MM-DD')
   - 例：`WHERE to_char(create_time, 'YYYY-MM-DD') >= '2025-01-01'`
   - 例：`WHERE to_char(create_date, 'YYYY-MM-DD') >= '2025-01-01'`
3. **JOIN 时两边类型必须一致**：如果连接列类型不同，给一边加显式 cast
   - 例：`a.user_id::text = b.uid`（如果 a.user_id 是 big_id，b.uid 是 int
5. **NULL 比较**：用 `IS NULL` / `IS NOT NULL`，不要用 `= NULL`
6. **时间计算**：如果字段类型为text类型，需要先将字段转化为timestamp类型
   - 例：create_time::timestamp + interval '1' day、create_time::timestamp + interval '1' month
7. **数值字段比较禁止用引号**：`WHERE id = 123` 而不是 `WHERE id = '123'`
8. **在计算率的时候**：不能直接除0，要case when 判断分母是否为0,且默认返回0。*1.0是为了确保结果是小数而不是整数*
  - 例：case when sum(score)>0 then sum(last_score)*1.0/sum(score) else 0 end as score_rate


⚠️ **判断依据**：在思考过程的第3步核对字段时，**留意每个字段的数据类型和注释中的格式描述**（例如注释里写 "格式: yyyy-mm-dd hh:mm:ss" 就用 `::timestamp`），看用法是否需要转换。

## 正确示例：
表结构：
表名: bi.db_course_student 表注释: 学生课程成绩表
列:
    student_id [bigint] (学生ID)
    grade [int] (成绩)
    create_time [text] (创建时间，格式: yyyy-mm-dd hh:mm:ss)
    birth_date [text] (出生日期，格式: yyyy-mm-dd)

✅ 正确 SQL：
  SELECT grade FROM bi.db_course_student WHERE student_id = 123
  SELECT * FROM bi.db_course_student WHERE create_time::timestamp >= '2025-01-01 00:00:00'   -- 含时分秒，用 timestamp
  SELECT * FROM bi.db_course_student WHERE birth_date::date >= '2000-01-01'                  -- 只有日期，用 date
  SELECT student_id::text FROM bi.db_course_student   -- 显式转 text

❌ 错误 SQL：
  SELECT db_course_student_grade FROM bi.db_course_student          -- 字段不存在
  WHERE student_id = '123'                                          -- bigint 跟字符串比较，Hologres 会报错
  WHERE create_time::date >= '2025-01-01'                           -- create_time 含时分秒，不能强转 date
  WHERE create_time >= '2025-01-01'                                 -- text 跟字符串比较看似 OK 但范围比较不可靠

## 表结构（只能使用以下列出的表和字段）：
{formatted_tables}

## 业务名词解释（必读，生成 SQL 前先理解这些业务术语）：
{formatted_glossary}

## 知识库：
{formatted_knowledge}""",
            user_message=f"问题：{nl_query}\n\n请先按格式输出思考过程，再输出最终SQL："
        )

        if content and len(content) > 3:
            cleaned_sql = extract_final_sql(str(content))
            if cleaned_sql:
                return cleaned_sql

        if content and content.strip().upper().startswith(('SELECT', 'WITH')):
            return content.strip()

        print(f"    ⚠️ AI返回无效内容: {repr(content[:100]) if content else 'None'}")
        return ""

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """调用 LLM API（带重试）"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            top_p=0.95,
            max_tokens=4000
        )
        if hasattr(completion, 'choices') and completion.choices:
            content = completion.choices[0].message.content
            print(f"    📝 AI原始返回: '{content[:200]}...' " if content and len(content) > 200 else f"    📝 AI原始返回: '{content}'")
            return content or ""
        raise ValueError('无法解析 AI 返回结果')

    def _split_tables(self, formatted_tables: str) -> List[str]:
        """将表结构拆分成多个批次"""

        table_pattern = r'(表名: [^\n]+(?:\n(?:    .+)?)+)'
        tables = re.findall(table_pattern, formatted_tables, re.MULTILINE)

        if not tables:
            return [formatted_tables[i:i+MAX_TABLE_LENGTH_PER_BATCH]
                    for i in range(0, len(formatted_tables), MAX_TABLE_LENGTH_PER_BATCH)]

        batches = []
        current_batch = []
        current_length = 0

        for table in tables:
            table_len = len(table)
            if not current_batch:
                current_batch.append(table)
                current_length = table_len
            elif current_length + table_len + 2 <= MAX_TABLE_LENGTH_PER_BATCH:
                current_batch.append(table)
                current_length += table_len + 2
            else:
                if len(current_batch) >= MIN_TABLES_PER_BATCH:
                    batches.append("\n\n".join(current_batch))
                    current_batch = [table]
                    current_length = table_len
                else:
                    current_batch.append(table)
                    current_length += table_len + 2
                    batches.append("\n\n".join(current_batch))
                    current_batch = []
                    current_length = 0

        if current_batch:
            batches.append("\n\n".join(current_batch))

        return batches

    def _merge_batches(self, batches: List[str], target_count: int) -> List[str]:
        """合并过多的批次到目标数量"""

        if len(batches) <= target_count:
            return batches

        avg_size = len(batches) // target_count
        merged = []

        for i in range(0, len(batches), avg_size):
            merged_batch = "\n\n".join(batches[i:i+avg_size])
            merged.append(merged_batch)
            if len(merged) >= target_count:
                break

        if len(merged) < len(batches):
            remaining = "\n\n".join(batches[len(merged)*avg_size:])
            merged[-1] = merged[-1] + "\n\n" + remaining

        return merged[:target_count]

    def _select_best_sql(self, candidates: List[Dict], query: str) -> str:
        """从多个候选SQL中选择最佳的一个"""

        if len(candidates) == 1:
            return candidates[0]['sql']

        keywords = set(re.findall(r'[一-龥a-zA-Z]+', query.lower()))

        for candidate in candidates:
            score = 0
            sql_lower = candidate['sql'].lower()

            for kw in keywords:
                if kw in sql_lower and len(kw) > 1:
                    score += 1

            score += (10 - candidate['batch']) * 0.1

            candidate['score'] = score

        best = max(candidates, key=lambda x: x['score'])
        print(f"    │   └─ 选择第{best['batch']}批的SQL（得分: {best['score']:.1f}）")

        return best['sql']

    def _format_knowledge(self, knowledge_json: List[Dict]) -> str:
        """格式化知识库示例"""
        if not knowledge_json:
            return "无可用知识库示例"
        examples = []
        for item in knowledge_json[:5]:
            question = item.get('question', '')
            sql = item.get('sql', '')
            if question and sql:
                examples.append(f"问题: {question}\nSQL: {sql}")
        return "\n\n".join(examples) if examples else "无可用示例"

    def _format_glossary(self, glossary: List[Dict]) -> str:
        """格式化业务名词解释"""
        if not glossary:
            return "无业务名词配置"
        items = []
        for item in glossary:
            term = item.get('term', '').strip()
            definition = item.get('definition', '').strip()
            if term and definition:
                items.append(f"- **{term}**：{definition}")
        return "\n".join(items) if items else "无业务名词配置"

# 本地MySQL数据库设置指南

## 概述
本系统现在支持将查询日志记录到本地MySQL数据库中，方便数据分析和系统监控。

## 环境变量配置

在 `.env` 文件中添加以下配置：

```bash
# 本地MySQL数据库配置（用于日志记录）
DB_LOCAL_HOST=localhost
DB_LOCAL_PORT=3306
DB_LOCAL_NAME=text_to_sql_logs
DB_LOCAL_USER=root
DB_LOCAL_PASSWORD=your_mysql_password_here
```

## 快速设置

### 1. 运行设置脚本
```bash
python setup_local_db.py
```

这个脚本会自动：
- 创建数据库 `text_to_sql_logs`
- 创建日志表 `query_logs`
- 测试数据库连接

### 2. 手动设置（如果自动设置失败）

#### 连接到MySQL
```bash
mysql -u root -p
```

#### 创建数据库
```sql
CREATE DATABASE text_to_sql_logs CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE text_to_sql_logs;
```

#### 创建日志表
```sql
CREATE TABLE query_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_query TEXT NOT NULL COMMENT '用户输入的自然语言查询',
    generated_sql TEXT COMMENT 'LLM生成的SQL语句',
    db_name VARCHAR(50) COMMENT '查询的数据库名称',
    prompt_content TEXT COMMENT '发送给LLM的prompt内容',
    model_name VARCHAR(100) COMMENT '使用的LLM模型名称',
    response_time_ms INT COMMENT '响应时长(毫秒)',
    token_count INT COMMENT '消耗的token数量',
    result_count INT COMMENT '查询结果行数',
    status ENUM('success', 'error') DEFAULT 'success' COMMENT '查询状态',
    error_message TEXT COMMENT '错误信息',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户查询日志表';
```

## 测试连接

运行测试脚本验证设置：
```bash
python test_logging.py
```

## 日志表字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INT | 自增主键 |
| user_query | TEXT | 用户输入的自然语言查询 |
| generated_sql | TEXT | LLM生成的SQL语句 |
| db_name | VARCHAR(50) | 查询的数据库名称 |
| library_name | VARCHAR(100) | 用户使用的库名称 |
| prompt_content | TEXT | 发送给LLM的prompt内容 |
| model_name | VARCHAR(100) | 使用的LLM模型名称 |
| response_time_ms | INT | 响应时长(毫秒) |
| token_count | INT | 消耗的token数量 |
| result_count | INT | 查询结果行数 |
| status | ENUM | 查询状态(success/error) |
| error_message | TEXT | 错误信息 |
| created_at | TIMESTAMP | 创建时间 |

## 常见问题

### 1. 连接失败
- 检查MySQL服务是否正在运行
- 验证用户名和密码是否正确
- 确认用户有创建数据库的权限

### 2. 权限问题
```sql
-- 为用户授予权限
GRANT ALL PRIVILEGES ON text_to_sql_logs.* TO 'your_user'@'localhost';
FLUSH PRIVILEGES;
```

### 3. 字符集问题
确保数据库和表使用 `utf8mb4` 字符集以支持中文字符。

## 查询示例

### 查看最近的查询
```sql
SELECT * FROM query_logs ORDER BY created_at DESC LIMIT 10;
```

### 统计查询成功率
```sql
SELECT 
    status,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM query_logs), 2) as percentage
FROM query_logs 
GROUP BY status;
```

### 分析响应时间
```sql
SELECT 
    AVG(response_time_ms) as avg_response_time,
    MAX(response_time_ms) as max_response_time,
    MIN(response_time_ms) as min_response_time
FROM query_logs 
WHERE status = 'success';
```

作者: Jamesenh 
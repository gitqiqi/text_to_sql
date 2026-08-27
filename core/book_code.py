# core/book_code.py - Git 仓库代码同步、函数/类级索引、混合召回、上下文扩展
import ast
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import text

from .db_manager import DatabasePoolManager
from .embedding_client import get_embedding_model, iter_embedding_models
from .utils import EMBEDDING_PROVIDER, _vectors_cache, monitor_function


DEFAULT_SYNC_HOUR = 1
DEFAULT_SYNC_MINUTE = 30
DEFAULT_CONTEXT_LINES = 18
DEFAULT_MAX_FILE_SIZE = 1_000_000
DEFAULT_MAX_SYMBOL_CHARS = 6000
DEFAULT_MAX_FILES = 5000
DEFAULT_PROMPT_ITEMS = 5
DEFAULT_VECTOR_LIMIT = 1500
DEFAULT_FULL_SCAN_CANDIDATES = 80

_monitor_thread = None
_monitor_lock = threading.Lock()
_sync_lock = threading.Lock()
_last_sync_result: Optional[Dict[str, Any]] = None

_DEFAULT_INCLUDE_EXTENSIONS = {
    '.py', '.pyi', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',
    '.java', '.kt', '.kts', '.scala', '.go', '.rs', '.rb', '.php',
    '.cs', '.c', '.cc', '.cpp', '.h', '.hpp', '.m', '.mm', '.swift',
    '.sql', '.sh', '.bash', '.zsh', '.md', '.rst', '.txt', '.yaml',
    '.yml', '.toml', '.ini', '.json', '.xml', '.html', '.css', '.scss',
    '.less', '.proto',
}
_DEFAULT_EXCLUDE_DIRS = {
    '.git', '.hg', '.svn', 'node_modules', 'dist', 'build', 'target',
    'vendor', '.venv', 'venv', '__pycache__', '.idea', '.vscode',
}


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {'0', 'false', 'no', 'off', ''}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _book_repo_path() -> Optional[Path]:
    raw = _env('BOOK_REPO_PATH')
    if not raw:
        return None
    return Path(raw).expanduser()


def _book_repo_name() -> str:
    raw = _env('BOOK_REPO_NAME')
    if raw:
        return raw
    repo_path = _book_repo_path()
    if repo_path:
        return repo_path.name or 'book'
    return 'book'


def _book_sync_enabled() -> bool:
    return _bool_env('BOOK_SYNC_ENABLED', False)


def _book_sync_hour() -> int:
    return max(0, min(23, _int_env('BOOK_SYNC_HOUR', DEFAULT_SYNC_HOUR)))


def _book_sync_minute() -> int:
    return max(0, min(59, _int_env('BOOK_SYNC_MINUTE', DEFAULT_SYNC_MINUTE)))


def _book_context_lines() -> int:
    return max(0, _int_env('BOOK_CONTEXT_LINES', DEFAULT_CONTEXT_LINES))


def _book_max_file_size() -> int:
    return max(50_000, _int_env('BOOK_MAX_FILE_SIZE', DEFAULT_MAX_FILE_SIZE))


def _book_max_symbol_chars() -> int:
    return max(1000, _int_env('BOOK_MAX_SYMBOL_CHARS', DEFAULT_MAX_SYMBOL_CHARS))


def _book_max_files() -> int:
    return max(1, _int_env('BOOK_MAX_FILES', DEFAULT_MAX_FILES))


def _book_prompt_items() -> int:
    return max(1, _int_env('BOOK_PROMPT_ITEMS', DEFAULT_PROMPT_ITEMS))


def _book_full_scan_candidates() -> int:
    return max(10, _int_env('BOOK_FULL_SCAN_CANDIDATES', DEFAULT_FULL_SCAN_CANDIDATES))


def _book_include_extensions() -> set[str]:
    raw = _env('BOOK_INCLUDE_EXTENSIONS')
    if raw:
        return {item.strip().lower() for item in raw.split(',') if item.strip()}
    return set(_DEFAULT_INCLUDE_EXTENSIONS)


def _book_exclude_dirs() -> set[str]:
    raw = _env('BOOK_EXCLUDE_DIRS')
    if raw:
        return {item.strip() for item in raw.split(',') if item.strip()}
    return set(_DEFAULT_EXCLUDE_DIRS)


def _book_repo_root() -> Optional[Path]:
    repo = _book_repo_path()
    if repo and repo.exists():
        return repo
    return None


def _json_default(value: Any):
    if isinstance(value, datetime):
        return value.isoformat(sep=' ', timespec='seconds')
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _truncate_middle(text_value: str, limit: int = DEFAULT_VECTOR_LIMIT) -> str:
    value = _normalize_text(text_value)
    if len(value) <= limit:
        return value
    head = max(200, limit // 2)
    tail = max(200, limit - head)
    return value[:head] + '\n...(中间已截断)...\n' + value[-tail:]


def _truncate_tail(text_value: str, limit: int) -> str:
    value = _normalize_text(text_value)
    if len(value) <= limit:
        return value
    return value[:limit] + '\n...(已截断)'


def _text_hash(value: str) -> str:
    return hashlib.md5(_normalize_text(value).encode('utf-8')).hexdigest()


def _source_hash(payload: Any) -> str:
    return hashlib.sha256(_json_dumps(payload).encode('utf-8')).hexdigest()


def _lines_slice(lines: Sequence[str], start_line: int, end_line: int) -> str:
    if not lines:
        return ''
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), max(start_idx, end_line))
    return '\n'.join(lines[start_idx:end_idx]).rstrip()


def _context_slice(lines: Sequence[str], start_line: int, end_line: int, padding: int) -> str:
    if not lines:
        return ''
    start_idx = max(0, start_line - 1 - padding)
    end_idx = min(len(lines), end_line + padding)
    return '\n'.join(lines[start_idx:end_idx]).rstrip()


def _relative_module_path(repo_root: Path, file_path: Path) -> str:
    try:
        rel = file_path.relative_to(repo_root)
    except Exception:
        rel = file_path.name
    rel_text = str(rel).replace('\\', '/')
    if '.' in rel_text:
        return rel_text.rsplit('.', 1)[0].replace('/', '.')
    return rel_text.replace('/', '.')


def _detect_language(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        '.py': 'python',
        '.pyi': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.mjs': 'javascript',
        '.cjs': 'javascript',
        '.java': 'java',
        '.kt': 'kotlin',
        '.kts': 'kotlin',
        '.scala': 'scala',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.cs': 'csharp',
        '.c': 'c',
        '.cc': 'cpp',
        '.cpp': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.m': 'objective-c',
        '.mm': 'objective-c++',
        '.swift': 'swift',
        '.sql': 'sql',
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.md': 'markdown',
        '.rst': 'rst',
        '.txt': 'text',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.less': 'less',
        '.proto': 'proto',
    }.get(suffix, suffix.lstrip('.') or 'text')


def _should_include_file(path: Path) -> bool:
    include_exts = _book_include_extensions()
    if path.suffix.lower() not in include_exts:
        return False
    parts = {part for part in path.parts}
    if parts & _book_exclude_dirs():
        return False
    return True


def _read_text_file(path: Path, max_size: int) -> Tuple[str, bool]:
    try:
        size = path.stat().st_size
    except Exception:
        size = None

    if size is not None and size > max_size:
        try:
            raw = path.read_bytes()[:max_size]
            return raw.decode('utf-8', errors='ignore'), True
        except Exception:
            return '', True

    encodings = ('utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin-1')
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding, errors='ignore'), False
        except Exception:
            continue
    return '', False


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ['git', '-C', str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _git_commit_hash(repo_root: Path) -> str:
    try:
        return _run_git(repo_root, 'rev-parse', 'HEAD')
    except Exception:
        return ''


def _git_branch_name(repo_root: Path) -> str:
    try:
        branch = _run_git(repo_root, 'rev-parse', '--abbrev-ref', 'HEAD')
        return branch or 'HEAD'
    except Exception:
        return 'HEAD'


def _git_remote_url(repo_root: Path) -> str:
    try:
        return _run_git(repo_root, 'remote', 'get-url', 'origin')
    except Exception:
        return ''


def _git_tracked_files(repo_root: Path) -> List[Path]:
    try:
        output = subprocess.run(
            ['git', '-C', str(repo_root), 'ls-files', '-z'],
            capture_output=True,
            check=True,
        ).stdout
    except Exception:
        output = b''
    items = [item for item in output.split(b'\0') if item]
    paths = []
    for item in items:
        try:
            rel = Path(item.decode('utf-8', errors='ignore'))
        except Exception:
            continue
        if rel.parts and not ({part for part in rel.parts} & _book_exclude_dirs()):
            paths.append(repo_root / rel)
    return paths[:_book_max_files()]


def _tokenize_query(query: str) -> List[str]:
    raw_tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z_][A-Za-z0-9_\.:-]*|\d+', query or '')
    tokens = []
    for token in raw_tokens:
        token = token.strip().strip('`"\'"')
        if token:
            tokens.append(token)
            if re.fullmatch(r'[\u4e00-\u9fff]{3,}', token):
                for size in (4, 3, 2):
                    if len(token) < size:
                        continue
                    for idx in range(0, len(token) - size + 1):
                        tokens.append(token[idx:idx + size])
    return list(dict.fromkeys(tokens))


def _normalize_book_search_mode(search_mode: Optional[str]) -> str:
    mode = (search_mode or '').strip().lower().replace('-', '_')
    if mode in {'all', 'full', 'full_scan', 'keyword', 'keywords'}:
        return 'all'
    if mode in {'selected', 'selected_table', 'exact'}:
        return 'selected_table'
    return 'vector'


def _join_query_hints(query: str, table_hints: Optional[Sequence[str]] = None) -> str:
    hints = []
    for hint in table_hints or []:
        value = _normalize_text(hint)
        if value:
            hints.append(value)
            if '.' in value:
                hints.append(value.split('.')[-1])
    hint_text = ' '.join(dict.fromkeys(hints))
    return f"{query}\n{hint_text}".strip() if hint_text else query


def _extract_call_names(node: ast.AST) -> List[str]:
    names: List[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return list(dict.fromkeys(names))


def _leading_comment_block(lines: Sequence[str], start_line: int) -> str:
    if start_line <= 1:
        return ''
    comments = []
    idx = start_line - 2
    while idx >= 0:
        line = lines[idx].rstrip()
        if not line:
            if comments:
                break
            idx -= 1
            continue
        stripped = line.lstrip()
        if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            comments.append(stripped)
            idx -= 1
            continue
        break
    comments.reverse()
    return '\n'.join(comments).strip()


def _python_signature(lines: Sequence[str], node: ast.AST) -> str:
    if not lines or not hasattr(node, 'lineno'):
        return ''
    idx = getattr(node, 'lineno', 1) - 1
    if idx < 0 or idx >= len(lines):
        return ''
    line = lines[idx].strip()
    return _truncate_tail(line, 300)


def _python_module_summary(lines: Sequence[str], docstring: str, imports: List[str], symbols: List[str]) -> str:
    parts = []
    if docstring:
        parts.append(f"模块说明: {docstring}")
    if imports:
        parts.append(f"导入: {', '.join(imports[:30])}")
    if symbols:
        parts.append(f"顶层符号: {', '.join(symbols[:40])}")
    if lines:
        preview = '\n'.join(lines[: min(len(lines), 120)])
        parts.append(f"代码预览:\n{preview}")
    return '\n'.join(parts).strip()


def _collect_python_imports(module: ast.Module) -> List[str]:
    items: List[str] = []
    for node in module.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                items.append(alias.name if not alias.asname else f"{alias.name} as {alias.asname}")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ''
            for alias in node.names:
                name = alias.name if not alias.asname else f"{alias.name} as {alias.asname}"
                items.append(f"from {module_name} import {name}".strip())
    return list(dict.fromkeys(items))


def _make_record(
    *,
    db_name: str,
    repo_name: str,
    repo_path: str,
    branch_name: str,
    commit_hash: str,
    file_path: str,
    file_name: str,
    module_path: str,
    language: str,
    symbol_type: str,
    symbol_name: str,
    qualified_name: str,
    parent_name: str = '',
    signature: str = '',
    docstring: str = '',
    leading_comments: str = '',
    code_text: str = '',
    context_text: str = '',
    imports_json: Optional[List[Any]] = None,
    references_json: Optional[List[Any]] = None,
    context_json: Optional[Dict[str, Any]] = None,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    symbol_order: int = 0,
    is_active: bool = True,
) -> Dict[str, Any]:
    file_text = _normalize_text(file_path)
    qualified = _normalize_text(qualified_name)
    code_text = _normalize_text(code_text)
    context_text = _normalize_text(context_text)
    signature = _normalize_text(signature)
    docstring = _normalize_text(docstring)
    leading_comments = _normalize_text(leading_comments)
    file_summary = _normalize_text(leading_comments)
    imports_json = imports_json or []
    references_json = references_json or []
    context_json = context_json or {}
    vector_parts = [
        f"仓库: {repo_name}",
        f"路径: {file_text}",
        f"语言: {language}",
        f"类型: {symbol_type}",
        f"符号: {qualified or symbol_name or file_name}",
        f"父级: {parent_name}" if parent_name else '',
        f"签名: {signature}" if signature else '',
        f"说明: {docstring}" if docstring else '',
        f"备注: {file_summary}" if file_summary and file_summary != docstring else '',
        f"导入: {', '.join(str(x) for x in imports_json[:20])}" if imports_json else '',
        f"引用: {', '.join(str(x) for x in references_json[:20])}" if references_json else '',
        f"上下文:\n{context_text}" if context_text else '',
        f"代码:\n{code_text}" if code_text else '',
    ]
    vector_text = '\n'.join(part for part in vector_parts if part)
    source_hash = _source_hash({
        'repo_name': repo_name,
        'repo_path': repo_path,
        'branch_name': branch_name,
        'commit_hash': commit_hash,
        'file_path': file_path,
        'symbol_type': symbol_type,
        'symbol_name': symbol_name,
        'qualified_name': qualified,
        'parent_name': parent_name,
        'signature': signature,
        'docstring': docstring,
        'leading_comments': leading_comments,
        'code_text': code_text,
        'context_text': context_text,
        'imports_json': imports_json,
        'references_json': references_json,
        'context_json': context_json,
        'line_start': line_start,
        'line_end': line_end,
    })
    return {
        'db_name': db_name,
        'repo_name': repo_name,
        'repo_path': repo_path,
        'branch_name': branch_name,
        'commit_hash': commit_hash,
        'file_path': file_path,
        'file_name': file_name,
        'module_path': module_path,
        'language': language,
        'symbol_type': symbol_type,
        'symbol_name': symbol_name,
        'qualified_name': qualified,
        'parent_name': parent_name,
        'signature': signature,
        'docstring': docstring,
        'leading_comments': leading_comments,
        'code_text': code_text,
        'context_text': context_text,
        'imports_json': imports_json,
        'references_json': references_json,
        'context_json': context_json,
        'line_start': line_start,
        'line_end': line_end,
        'symbol_order': symbol_order,
        'vector_text': vector_text,
        'text_hash': _text_hash(vector_text),
        'source_hash': source_hash,
        'is_active': is_active,
        'last_seen_at': datetime.now(),
    }


def _python_file_records(
    *,
    db_name: str,
    repo_name: str,
    repo_path: str,
    branch_name: str,
    commit_hash: str,
    repo_root: Path,
    file_path: Path,
    content: str,
) -> List[Dict[str, Any]]:
    rel_path = str(file_path.relative_to(repo_root)).replace('\\', '/')
    module_path = _relative_module_path(repo_root, file_path)
    file_name = file_path.name
    lines = content.splitlines()
    try:
        module = ast.parse(content)
    except Exception:
        return [_make_record(
            db_name=db_name,
            repo_name=repo_name,
            repo_path=repo_path,
            branch_name=branch_name,
            commit_hash=commit_hash,
            file_path=rel_path,
            file_name=file_name,
            module_path=module_path,
            language='python',
            symbol_type='file',
            symbol_name=file_name,
            qualified_name='',
            signature='',
            docstring='',
            leading_comments='',
            code_text=_truncate_middle(content, _book_max_symbol_chars()),
            context_text=_truncate_middle(content, _book_max_symbol_chars()),
            imports_json=[],
            references_json=[],
            context_json={'kind': 'file', 'truncated': True},
            symbol_order=0,
        )]

    module_doc = ast.get_docstring(module, clean=False) or ''
    imports = _collect_python_imports(module)
    symbols: List[str] = []
    records: List[Dict[str, Any]] = []
    context_lines = _book_context_lines()

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope: List[Tuple[str, str]] = []

        def _emit(self, node: ast.AST, symbol_type: str, symbol_name: str):
            start_line = getattr(node, 'lineno', None)
            end_line = getattr(node, 'end_lineno', None) or start_line
            if not start_line or not end_line:
                return
            parent_names = [name for name, _ in self.scope]
            parent_name = '.'.join(parent_names)
            qualified = '.'.join(parent_names + [symbol_name]) if parent_names else symbol_name
            code_text = _lines_slice(lines, start_line, end_line)
            context_text = _context_slice(lines, start_line, end_line, context_lines)
            docstring = ast.get_docstring(node, clean=False) or ''
            leading_comments = _leading_comment_block(lines, start_line)
            signature = _python_signature(lines, node)
            references = _extract_call_names(node)
            records.append(_make_record(
                db_name=db_name,
                repo_name=repo_name,
                repo_path=repo_path,
                branch_name=branch_name,
                commit_hash=commit_hash,
                file_path=rel_path,
                file_name=file_name,
                module_path=module_path,
                language='python',
                symbol_type=symbol_type,
                symbol_name=symbol_name,
                qualified_name=qualified,
                parent_name=parent_name,
                signature=signature,
                docstring=docstring,
                leading_comments=leading_comments,
                code_text=_truncate_middle(code_text, _book_max_symbol_chars()),
                context_text=_truncate_middle(context_text, _book_max_symbol_chars()),
                imports_json=imports,
                references_json=references,
                context_json={'kind': symbol_type, 'start_line': start_line, 'end_line': end_line},
                line_start=start_line,
                line_end=end_line,
                symbol_order=start_line,
            ))
            symbols.append(qualified)

        def visit_ClassDef(self, node: ast.ClassDef):
            self._emit(node, 'class', node.name)
            self.scope.append((node.name, 'class'))
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            if any(kind == 'function' for _, kind in self.scope):
                return
            symbol_type = 'method' if self.scope and self.scope[-1][1] == 'class' else 'function'
            self._emit(node, symbol_type, node.name)
            self.scope.append((node.name, 'function'))
            self.generic_visit(node)
            self.scope.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self.visit_FunctionDef(node)  # type: ignore[arg-type]

    Visitor().visit(module)
    module_summary = _python_module_summary(lines, module_doc, imports, symbols)
    records.insert(0, _make_record(
        db_name=db_name,
        repo_name=repo_name,
        repo_path=repo_path,
        branch_name=branch_name,
        commit_hash=commit_hash,
        file_path=rel_path,
        file_name=file_name,
        module_path=module_path,
        language='python',
        symbol_type='file',
        symbol_name=file_name,
        qualified_name='',
        signature='module',
        docstring=module_doc,
        leading_comments=_leading_comment_block(lines, 1),
        code_text=_truncate_middle(content, _book_max_symbol_chars()),
        context_text=_truncate_middle(module_summary, _book_max_symbol_chars()),
        imports_json=imports,
        references_json=symbols,
        context_json={'kind': 'file', 'symbols': symbols, 'imports': imports},
        line_start=1,
        line_end=len(lines) or 1,
        symbol_order=0,
    ))
    return records


_GENERIC_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ('class', re.compile(r'^\s*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b')),
    ('function', re.compile(r'^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')),
    ('function', re.compile(r'^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>)')),
    ('method', re.compile(r'^\s*(?:public|private|protected|static|final|native|synchronized|override|async|\s)+[A-Za-z0-9_<>,\[\]\s]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{')),
]


def _brace_block_end(lines: Sequence[str], start_idx: int) -> int:
    brace_count = 0
    started = False
    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        for ch in line:
            if ch == '{':
                brace_count += 1
                started = True
            elif ch == '}':
                brace_count -= 1
        if started and brace_count <= 0:
            return idx + 1
    return len(lines)


def _generic_file_records(
    *,
    db_name: str,
    repo_name: str,
    repo_path: str,
    branch_name: str,
    commit_hash: str,
    repo_root: Path,
    file_path: Path,
    content: str,
) -> List[Dict[str, Any]]:
    rel_path = str(file_path.relative_to(repo_root)).replace('\\', '/')
    module_path = _relative_module_path(repo_root, file_path)
    language = _detect_language(file_path)
    file_name = file_path.name
    lines = content.splitlines()
    max_chars = _book_max_symbol_chars()
    records: List[Dict[str, Any]] = []
    matched_ranges: List[Tuple[int, int]] = []

    for symbol_type, pattern in _GENERIC_PATTERNS:
        for idx, line in enumerate(lines):
            match = pattern.match(line)
            if not match:
                continue
            symbol_name = match.group(1)
            start_line = idx + 1
            end_line = _brace_block_end(lines, idx)
            if any(start_line >= s and start_line <= e for s, e in matched_ranges):
                continue
            matched_ranges.append((start_line, end_line))
            code_text = _lines_slice(lines, start_line, end_line)
            context_text = _context_slice(lines, start_line, end_line, _book_context_lines())
            leading_comments = _leading_comment_block(lines, start_line)
            records.append(_make_record(
                db_name=db_name,
                repo_name=repo_name,
                repo_path=repo_path,
                branch_name=branch_name,
                commit_hash=commit_hash,
                file_path=rel_path,
                file_name=file_name,
                module_path=module_path,
                language=language,
                symbol_type=symbol_type,
                symbol_name=symbol_name,
                qualified_name=symbol_name,
                parent_name='',
                signature=_truncate_tail(line.strip(), 300),
                docstring='',
                leading_comments=leading_comments,
                code_text=_truncate_middle(code_text, max_chars),
                context_text=_truncate_middle(context_text, max_chars),
                imports_json=[],
                references_json=[],
                context_json={'kind': symbol_type, 'start_line': start_line, 'end_line': end_line},
                line_start=start_line,
                line_end=end_line,
                symbol_order=start_line,
            ))

    file_summary = _truncate_middle(content, max_chars)
    records.insert(0, _make_record(
        db_name=db_name,
        repo_name=repo_name,
        repo_path=repo_path,
        branch_name=branch_name,
        commit_hash=commit_hash,
        file_path=rel_path,
        file_name=file_name,
        module_path=module_path,
        language=language,
        symbol_type='file',
        symbol_name=file_name,
        qualified_name='',
        signature='module',
        docstring='',
        leading_comments=_leading_comment_block(lines, 1),
        code_text=file_summary,
        context_text=file_summary,
        imports_json=[],
        references_json=[record['symbol_name'] for record in records[1:]],
        context_json={'kind': 'file', 'symbols': [record['symbol_name'] for record in records[1:]]},
        line_start=1,
        line_end=len(lines) or 1,
        symbol_order=0,
    ))
    return records


def _build_records_for_file(
    *,
    db_name: str,
    repo_name: str,
    repo_path: str,
    branch_name: str,
    commit_hash: str,
    repo_root: Path,
    file_path: Path,
) -> List[Dict[str, Any]]:
    content, truncated = _read_text_file(file_path, _book_max_file_size())
    if not content:
        return []

    language = _detect_language(file_path)
    if language == 'python' and not truncated:
        return _python_file_records(
            db_name=db_name,
            repo_name=repo_name,
            repo_path=repo_path,
            branch_name=branch_name,
            commit_hash=commit_hash,
            repo_root=repo_root,
            file_path=file_path,
            content=content,
        )

    if truncated:
        content = _truncate_middle(content, _book_max_symbol_chars())

    return _generic_file_records(
        db_name=db_name,
        repo_name=repo_name,
        repo_path=repo_path,
        branch_name=branch_name,
        commit_hash=commit_hash,
        repo_root=repo_root,
        file_path=file_path,
        content=content,
    )


class BookKnowledgeStore:
    """仓库代码知识表：函数/类/文件级混合索引。"""

    _sql_path_disabled = set()
    _sql_path_lock = threading.Lock()
    _schema_ready = set()
    _schema_lock = threading.Lock()

    def __init__(self, db_name: Optional[str] = None, repo_path: Optional[str] = None,
                 repo_name: Optional[str] = None, ensure_schema: bool = False):
        self.db_name = db_name or os.getenv('AUTH_DB_NAME', os.getenv('APP_AUTH_DB_NAME', 'hologres'))
        self.repo_path = Path(repo_path).expanduser() if repo_path else _book_repo_path()
        self.repo_name = repo_name or _book_repo_name()
        self.engine = DatabasePoolManager.get_engine(self.db_name)
        self.cache_prefix = f"book_code:{self.db_name}:{self.repo_name}"
        self.ensure_schema = ensure_schema
        if self.ensure_schema:
            self._ensure_schema()

    @staticmethod
    def _embedding_col(provider: str) -> str:
        return 'doubao_embedding' if provider == 'api' else 'local_embedding'

    @staticmethod
    def _embedding_provider_from_model(model) -> str:
        name = getattr(model, 'name', '')
        return 'api' if str(name).startswith('api:') else 'local'

    def _schema_is_ready(self) -> bool:
        if self.db_name in self._schema_ready:
            return True
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text("""
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'knowledge'
                      AND table_name = 'book_code_knowledge'
                    LIMIT 1
                """)).fetchone()
        except Exception:
            return False
        ready = row is not None
        if ready:
            with self._schema_lock:
                self._schema_ready.add(self.db_name)
        return ready

    def _ensure_schema(self):
        if self._schema_is_ready():
            return

        with self._schema_lock:
            if self._schema_is_ready():
                return

        ddl_statements = [
            "CREATE SCHEMA IF NOT EXISTS knowledge",
            """
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
                local_embedding REAL[] CHECK (array_ndims(local_embedding) = 1 AND array_length(local_embedding, 1) = 384),
                doubao_embedding REAL[] CHECK (array_ndims(doubao_embedding) = 1 AND array_length(doubao_embedding, 1) = 2048),
                text_hash VARCHAR(64),
                source_hash VARCHAR(64),
                is_active BOOLEAN DEFAULT TRUE,
                last_seen_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS repo_name TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS repo_path TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS branch_name TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS commit_hash TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS file_path TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS file_name TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS module_path TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS language TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS symbol_type TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS symbol_name TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS qualified_name TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS parent_name TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS signature TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS docstring TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS leading_comments TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS code_text TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS context_text TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS imports_json JSONB",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS references_json JSONB",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS context_json JSONB",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS line_start INTEGER",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS line_end INTEGER",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS symbol_order INTEGER",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS vector_text TEXT",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS local_embedding REAL[]",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS doubao_embedding REAL[]",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS text_hash VARCHAR(64)",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS source_hash VARCHAR(64)",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMP",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()",
            "ALTER TABLE knowledge.book_code_knowledge ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()",
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_book_code_knowledge_key
            ON knowledge.book_code_knowledge (db_name, repo_name, file_path, symbol_type, qualified_name)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_book_code_knowledge_active
            ON knowledge.book_code_knowledge (db_name, repo_name, is_active, updated_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_book_code_knowledge_file
            ON knowledge.book_code_knowledge (db_name, repo_name, file_path, symbol_order)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_book_code_knowledge_symbol
            ON knowledge.book_code_knowledge (db_name, repo_name, symbol_name)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_book_code_knowledge_commit
            ON knowledge.book_code_knowledge (db_name, repo_name, commit_hash)
            """,
        ]

        with self.engine.connect() as conn:
            for ddl in ddl_statements:
                conn.execute(text(ddl))
            conn.commit()

        with self._schema_lock:
            self._schema_ready.add(self.db_name)

    def _existing_snapshot(self, embedding_col: str) -> Dict[str, Tuple[Optional[str], Optional[int], bool]]:
        snapshot = {}
        with self.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT file_path, symbol_type, qualified_name, text_hash, id, {embedding_col} IS NOT NULL AS has_col
                FROM knowledge.book_code_knowledge
                WHERE db_name = :db_name AND repo_name = :repo_name
            """), {
                'db_name': self.db_name,
                'repo_name': self.repo_name,
            }).fetchall()
        for row in rows:
            key = f"{row[0]}|{row[1]}|{row[2]}"
            snapshot[key] = (row[3], row[4], bool(row[5]))
        return snapshot

    def _upsert_rows(self, conn, embedding_col: str, rows: List[Dict[str, Any]], embeddings: List[np.ndarray]) -> None:
        if not rows:
            return
        params = []
        for row, embedding in zip(rows, embeddings):
            params.append({
                'id': row.get('_row_id'),
                'db_name': row['db_name'],
                'repo_name': row['repo_name'],
                'repo_path': row.get('repo_path'),
                'branch_name': row.get('branch_name'),
                'commit_hash': row.get('commit_hash'),
                'file_path': row['file_path'],
                'file_name': row.get('file_name'),
                'module_path': row.get('module_path'),
                'language': row.get('language'),
                'symbol_type': row['symbol_type'],
                'symbol_name': row.get('symbol_name') or '',
                'qualified_name': row.get('qualified_name') or '',
                'parent_name': row.get('parent_name'),
                'signature': row.get('signature'),
                'docstring': row.get('docstring'),
                'leading_comments': row.get('leading_comments'),
                'code_text': row.get('code_text'),
                'context_text': row.get('context_text'),
                'imports_json': _json_dumps(row.get('imports_json') or []),
                'references_json': _json_dumps(row.get('references_json') or []),
                'context_json': _json_dumps(row.get('context_json') or {}),
                'line_start': row.get('line_start'),
                'line_end': row.get('line_end'),
                'symbol_order': row.get('symbol_order', 0),
                'vector_text': row.get('vector_text'),
                'embedding': embedding.tolist(),
                'text_hash': row.get('text_hash'),
                'source_hash': row.get('source_hash'),
            })

        update_sql = f"""
            UPDATE knowledge.book_code_knowledge
            SET repo_path = :repo_path,
                branch_name = :branch_name,
                commit_hash = :commit_hash,
                file_name = :file_name,
                module_path = :module_path,
                language = :language,
                symbol_name = :symbol_name,
                qualified_name = :qualified_name,
                parent_name = :parent_name,
                signature = :signature,
                docstring = :docstring,
                leading_comments = :leading_comments,
                code_text = :code_text,
                context_text = :context_text,
                imports_json = CAST(:imports_json AS jsonb),
                references_json = CAST(:references_json AS jsonb),
                context_json = CAST(:context_json AS jsonb),
                line_start = :line_start,
                line_end = :line_end,
                symbol_order = :symbol_order,
                vector_text = :vector_text,
                {embedding_col} = :embedding,
                text_hash = :text_hash,
                source_hash = :source_hash,
                last_seen_at = NOW(),
                is_active = TRUE,
                updated_at = NOW()
            WHERE id = :id
        """
        insert_sql = f"""
            INSERT INTO knowledge.book_code_knowledge (
                db_name, repo_name, repo_path, branch_name, commit_hash,
                file_path, file_name, module_path, language,
                symbol_type, symbol_name, qualified_name, parent_name,
                signature, docstring, leading_comments, code_text, context_text,
                imports_json, references_json, context_json,
                line_start, line_end, symbol_order, vector_text,
                {embedding_col}, text_hash, source_hash, last_seen_at, is_active, updated_at
            )
            VALUES (
                :db_name, :repo_name, :repo_path, :branch_name, :commit_hash,
                :file_path, :file_name, :module_path, :language,
                :symbol_type, :symbol_name, :qualified_name, :parent_name,
                :signature, :docstring, :leading_comments, :code_text, :context_text,
                CAST(:imports_json AS jsonb), CAST(:references_json AS jsonb), CAST(:context_json AS jsonb),
                :line_start, :line_end, :symbol_order, :vector_text,
                :embedding, :text_hash, :source_hash, NOW(), TRUE, NOW()
            )
        """
        update_params = [dict(item, id=item['id']) for item in params if item.get('id') is not None]
        if update_params:
            conn.execute(text(update_sql), update_params)
        insert_params = [item for item in params if item.get('id') is None]
        if insert_params:
            conn.execute(text(insert_sql), insert_params)

    def save_records_incrementally(self, model, records: List[Dict[str, Any]],
                                  vector_texts: List[str]) -> Dict[str, int]:
        self._ensure_schema()
        embedding_col = self._embedding_col(self._embedding_provider_from_model(model))
        stats = {'new': 0, 'changed': 0, 'unchanged': 0, 'removed': 0}
        if not records:
            return stats

        existing = self._existing_snapshot(embedding_col)
        new_rows: List[Dict[str, Any]] = []
        new_texts: List[str] = []
        changed_rows: List[Dict[str, Any]] = []
        changed_texts: List[str] = []
        current_keys = set()

        for record, vector_text in zip(records, vector_texts):
            key = f"{record['file_path']}|{record['symbol_type']}|{record.get('qualified_name') or ''}"
            current_keys.add(key)
            record['vector_text'] = vector_text
            record['text_hash'] = _text_hash(vector_text)
            old_hash, row_id, has_col = existing.get(key, (None, None, False))
            if row_id is None:
                new_rows.append(record)
                new_texts.append(vector_text)
            elif old_hash != record['text_hash'] or not has_col:
                record['_row_id'] = row_id
                changed_rows.append(record)
                changed_texts.append(vector_text)

        removed_keys = set(existing.keys()) - current_keys
        stats['new'] = len(new_rows)
        stats['changed'] = len(changed_rows)
        stats['unchanged'] = len(records) - stats['new'] - stats['changed']
        stats['removed'] = len(removed_keys)

        if stats['new'] == 0 and stats['changed'] == 0 and stats['removed'] == 0:
            print(f"   ✅ [book] {self.repo_name} 已是最新")
            return stats

        embeddings_list: List[np.ndarray] = []
        to_encode_texts = new_texts + changed_texts
        if to_encode_texts:
            print(f"   ├─ 生成 {len(to_encode_texts)} 个仓库代码向量...")
            batch_size = 50
            for i in range(0, len(to_encode_texts), batch_size):
                batch = to_encode_texts[i:i + batch_size]
                batch_embeddings = model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                embeddings_list.extend(batch_embeddings)

        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                for key in removed_keys:
                    file_path, symbol_type, qualified_name = key.split('|', 2)
                    conn.execute(text(f"""
                        UPDATE knowledge.book_code_knowledge
                        SET {embedding_col} = NULL,
                            is_active = FALSE,
                            updated_at = NOW()
                        WHERE db_name = :db_name
                          AND repo_name = :repo_name
                          AND file_path = :file_path
                          AND symbol_type = :symbol_type
                          AND qualified_name = :qualified_name
                    """), {
                        'db_name': self.db_name,
                        'repo_name': self.repo_name,
                        'file_path': file_path,
                        'symbol_type': symbol_type,
                        'qualified_name': qualified_name,
                    })

                changed_with_id = [r for r in changed_rows if '_row_id' in r]
                if changed_with_id:
                    changed_embeddings = embeddings_list[len(new_rows):]
                    update_params = []
                    for record, embedding in zip(changed_with_id, changed_embeddings):
                        update_params.append({
                            'id': record['_row_id'],
                            'repo_path': record.get('repo_path'),
                            'branch_name': record.get('branch_name'),
                            'commit_hash': record.get('commit_hash'),
                            'file_name': record.get('file_name'),
                            'module_path': record.get('module_path'),
                            'language': record.get('language'),
                            'symbol_name': record.get('symbol_name') or '',
                            'qualified_name': record.get('qualified_name') or '',
                            'parent_name': record.get('parent_name'),
                            'signature': record.get('signature'),
                            'docstring': record.get('docstring'),
                            'leading_comments': record.get('leading_comments'),
                            'code_text': record.get('code_text'),
                            'context_text': record.get('context_text'),
                            'imports_json': _json_dumps(record.get('imports_json') or []),
                            'references_json': _json_dumps(record.get('references_json') or []),
                            'context_json': _json_dumps(record.get('context_json') or {}),
                            'line_start': record.get('line_start'),
                            'line_end': record.get('line_end'),
                            'symbol_order': record.get('symbol_order', 0),
                            'vector_text': record.get('vector_text'),
                            'embedding': embedding.tolist(),
                            'text_hash': record.get('text_hash'),
                            'source_hash': record.get('source_hash'),
                        })
                    conn.execute(text(f"""
                        UPDATE knowledge.book_code_knowledge
                        SET repo_path = :repo_path,
                            branch_name = :branch_name,
                            commit_hash = :commit_hash,
                            file_name = :file_name,
                            module_path = :module_path,
                            language = :language,
                            symbol_name = :symbol_name,
                            qualified_name = :qualified_name,
                            parent_name = :parent_name,
                            signature = :signature,
                            docstring = :docstring,
                            leading_comments = :leading_comments,
                            code_text = :code_text,
                            context_text = :context_text,
                            imports_json = CAST(:imports_json AS jsonb),
                            references_json = CAST(:references_json AS jsonb),
                            context_json = CAST(:context_json AS jsonb),
                            line_start = :line_start,
                            line_end = :line_end,
                            symbol_order = :symbol_order,
                            vector_text = :vector_text,
                            {embedding_col} = :embedding,
                            text_hash = :text_hash,
                            source_hash = :source_hash,
                            last_seen_at = NOW(),
                            is_active = TRUE,
                            updated_at = NOW()
                        WHERE id = :id
                    """), update_params)

                if new_rows:
                    insert_params = []
                    for record, embedding in zip(new_rows, embeddings_list[:len(new_rows)]):
                        insert_params.append({
                            'db_name': record['db_name'],
                            'repo_name': record['repo_name'],
                            'repo_path': record.get('repo_path'),
                            'branch_name': record.get('branch_name'),
                            'commit_hash': record.get('commit_hash'),
                            'file_path': record['file_path'],
                            'file_name': record.get('file_name'),
                            'module_path': record.get('module_path'),
                            'language': record.get('language'),
                            'symbol_type': record['symbol_type'],
                            'symbol_name': record.get('symbol_name') or '',
                            'qualified_name': record.get('qualified_name') or '',
                            'parent_name': record.get('parent_name'),
                            'signature': record.get('signature'),
                            'docstring': record.get('docstring'),
                            'leading_comments': record.get('leading_comments'),
                            'code_text': record.get('code_text'),
                            'context_text': record.get('context_text'),
                            'imports_json': _json_dumps(record.get('imports_json') or []),
                            'references_json': _json_dumps(record.get('references_json') or []),
                            'context_json': _json_dumps(record.get('context_json') or {}),
                            'line_start': record.get('line_start'),
                            'line_end': record.get('line_end'),
                            'symbol_order': record.get('symbol_order', 0),
                            'vector_text': record.get('vector_text'),
                            'embedding': embedding.tolist(),
                            'text_hash': record.get('text_hash'),
                            'source_hash': record.get('source_hash'),
                        })
                    conn.execute(text(f"""
                        INSERT INTO knowledge.book_code_knowledge (
                            db_name, repo_name, repo_path, branch_name, commit_hash,
                            file_path, file_name, module_path, language,
                            symbol_type, symbol_name, qualified_name, parent_name,
                            signature, docstring, leading_comments, code_text, context_text,
                            imports_json, references_json, context_json,
                            line_start, line_end, symbol_order, vector_text,
                            {embedding_col}, text_hash, source_hash, last_seen_at, is_active, updated_at
                        )
                        VALUES (
                            :db_name, :repo_name, :repo_path, :branch_name, :commit_hash,
                            :file_path, :file_name, :module_path, :language,
                            :symbol_type, :symbol_name, :qualified_name, :parent_name,
                            :signature, :docstring, :leading_comments, :code_text, :context_text,
                            CAST(:imports_json AS jsonb), CAST(:references_json AS jsonb), CAST(:context_json AS jsonb),
                            :line_start, :line_end, :symbol_order, :vector_text,
                            :embedding, :text_hash, :source_hash, NOW(), TRUE, NOW()
                        )
                    """), insert_params)

                trans.commit()
                _vectors_cache.invalidate(f"{self.cache_prefix}:{embedding_col}")
                print(f"   ✅ [book] 批量保存 {len(records)} 条知识到 Hologres")
                return stats
            except Exception as exc:
                trans.rollback()
                print(f"   ❌ [book] 保存仓库代码知识失败: {exc}")
                raise

    def _load_all_vectors_cached(self, embedding_col: str = 'local_embedding'):
        cache_key = f"{self.cache_prefix}:{embedding_col}"
        cached = _vectors_cache.get(cache_key)
        if cached is not None:
            return cached

        sql = f"""
        SELECT repo_name, file_path, file_name, module_path, language, symbol_type, symbol_name,
               qualified_name, parent_name, signature, docstring, leading_comments,
               code_text, context_text, imports_json, references_json, context_json,
               line_start, line_end, symbol_order, vector_text, {embedding_col}
        FROM knowledge.book_code_knowledge
        WHERE db_name = :db_name
          AND repo_name = :repo_name
          AND {embedding_col} IS NOT NULL
          AND is_active IS TRUE
        """
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(sql), {
                    'db_name': self.db_name,
                    'repo_name': self.repo_name,
                }).fetchall()
        except Exception as exc:
            print(f"   ❌ [book] 加载仓库代码向量失败: {exc}")
            return [], np.array([])

        if not rows:
            return [], np.array([])

        records = []
        embeddings = []
        for row in rows:
            records.append({
                'repo_name': row[0],
                'file_path': row[1],
                'file_name': row[2],
                'module_path': row[3],
                'language': row[4],
                'symbol_type': row[5],
                'symbol_name': row[6],
                'qualified_name': row[7],
                'parent_name': row[8],
                'signature': row[9],
                'docstring': row[10],
                'leading_comments': row[11],
                'code_text': row[12],
                'context_text': row[13],
                'imports_json': row[14] if row[14] else [],
                'references_json': row[15] if row[15] else [],
                'context_json': row[16] if row[16] else {},
                'line_start': row[17],
                'line_end': row[18],
                'symbol_order': row[19],
                'vector_text': row[20] or '',
            })
            embeddings.append(row[21])

        result = (records, np.array(embeddings, dtype=np.float32))
        _vectors_cache.set(cache_key, result)
        print(f"   ├─ 加载 {len(records)} 条仓库代码向量到内存（{embedding_col}）")
        return result

    def _keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        tokens = _tokenize_query(query)
        if not tokens:
            return []

        clauses = []
        params: Dict[str, Any] = {
            'db_name': self.db_name,
            'repo_name': self.repo_name,
            'exact_query': query.strip().lower(),
        }
        for idx, token in enumerate(tokens[:8]):
            param_name = f"token_{idx}"
            params[param_name] = f"%{token}%"
            clauses.append(
                f"(LOWER(COALESCE(qualified_name, '')) LIKE LOWER(:{param_name}) "
                f"OR LOWER(COALESCE(symbol_name, '')) LIKE LOWER(:{param_name}) "
                f"OR LOWER(COALESCE(file_path, '')) LIKE LOWER(:{param_name}) "
                f"OR LOWER(COALESCE(docstring, '')) LIKE LOWER(:{param_name}) "
                f"OR LOWER(COALESCE(leading_comments, '')) LIKE LOWER(:{param_name}) "
                f"OR LOWER(COALESCE(context_text, '')) LIKE LOWER(:{param_name}) "
                f"OR LOWER(COALESCE(vector_text, '')) LIKE LOWER(:{param_name}) "
                f"OR LOWER(COALESCE(code_text, '')) LIKE LOWER(:{param_name}))"
            )
        score_parts = []
        for idx in range(min(len(tokens), 8)):
            param_name = f"token_{idx}"
            score_parts.extend([
                f"CASE WHEN LOWER(COALESCE(qualified_name, '')) LIKE LOWER(:{param_name}) THEN 30 ELSE 0 END",
                f"CASE WHEN LOWER(COALESCE(symbol_name, '')) LIKE LOWER(:{param_name}) THEN 24 ELSE 0 END",
                f"CASE WHEN LOWER(COALESCE(file_path, '')) LIKE LOWER(:{param_name}) THEN 16 ELSE 0 END",
                f"CASE WHEN LOWER(COALESCE(docstring, '')) LIKE LOWER(:{param_name}) THEN 12 ELSE 0 END",
                f"CASE WHEN LOWER(COALESCE(leading_comments, '')) LIKE LOWER(:{param_name}) THEN 10 ELSE 0 END",
                f"CASE WHEN LOWER(COALESCE(context_text, '')) LIKE LOWER(:{param_name}) THEN 7 ELSE 0 END",
                f"CASE WHEN LOWER(COALESCE(vector_text, '')) LIKE LOWER(:{param_name}) THEN 7 ELSE 0 END",
                f"CASE WHEN LOWER(COALESCE(code_text, '')) LIKE LOWER(:{param_name}) THEN 5 ELSE 0 END",
            ])
        keyword_score = ' + '.join(score_parts) if score_parts else '0'

        sql = f"""
        SELECT id, repo_name, repo_path, branch_name, commit_hash, file_path, file_name, module_path,
               language, symbol_type, symbol_name, qualified_name, parent_name, signature,
               docstring, leading_comments, code_text, context_text, imports_json,
               references_json, context_json, line_start, line_end, symbol_order, vector_text,
               text_hash, source_hash
        FROM knowledge.book_code_knowledge
        WHERE db_name = :db_name
          AND repo_name = :repo_name
          AND is_active IS TRUE
          AND ({' OR '.join(clauses)})
        ORDER BY
            CASE WHEN LOWER(COALESCE(qualified_name, '')) = :exact_query THEN 100 ELSE 0 END DESC,
            CASE WHEN LOWER(COALESCE(symbol_name, '')) = :exact_query THEN 80 ELSE 0 END DESC,
            ({keyword_score}) DESC,
            CASE WHEN symbol_type IN ('function', 'method', 'class') THEN 1 ELSE 0 END DESC,
            symbol_order ASC,
            updated_at DESC
        LIMIT :limit
        """
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(sql), {**params, 'limit': top_k}).fetchall()
        except Exception as exc:
            print(f"   ⚠️ [book] 关键词检索失败: {exc}")
            return []

        results = []
        for rank, row in enumerate(rows, 1):
            item = {
                'id': row[0],
                'repo_name': row[1],
                'repo_path': row[2],
                'branch_name': row[3],
                'commit_hash': row[4],
                'file_path': row[5],
                'file_name': row[6],
                'module_path': row[7],
                'language': row[8],
                'symbol_type': row[9],
                'symbol_name': row[10],
                'qualified_name': row[11],
                'parent_name': row[12],
                'signature': row[13],
                'docstring': row[14],
                'leading_comments': row[15],
                'code_text': row[16],
                'context_text': row[17],
                'imports_json': row[18] if row[18] else [],
                'references_json': row[19] if row[19] else [],
                'context_json': row[20] if row[20] else {},
                'line_start': row[21],
                'line_end': row[22],
                'symbol_order': row[23],
                'vector_text': row[24],
                'text_hash': row[25],
                'source_hash': row[26],
                '_keyword_rank': rank,
            }
            results.append(item)
        return results

    def _vector_search_via_holo_sql(self, query_embedding: List[float], top_k: int,
                                    embedding_col: str = 'local_embedding') -> List[Dict[str, Any]]:
        embedding_str = '{' + ','.join(str(x) for x in query_embedding) + '}'
        params = {
            'db_name': self.db_name,
            'repo_name': self.repo_name,
            'query_embedding': embedding_str,
            'top_k': top_k,
        }
        sql = f"""
        SELECT
            id, repo_name, repo_path, branch_name, commit_hash, file_path, file_name, module_path,
            language, symbol_type, symbol_name, qualified_name, parent_name, signature,
            docstring, leading_comments, code_text, context_text, imports_json,
            references_json, context_json, line_start, line_end, symbol_order, vector_text,
            pm_approx_squared_euclidean_distance({embedding_col}, CAST(:query_embedding AS float4[])) AS similarity_score
        FROM knowledge.book_code_knowledge
        WHERE db_name = :db_name
          AND repo_name = :repo_name
          AND {embedding_col} IS NOT NULL
          AND is_active IS TRUE
        ORDER BY similarity_score ASC
        LIMIT :top_k
        """
        with self.engine.connect() as conn:
            try:
                conn.execute(text("SET hg_computing_resource = 'serverless'"))
            except Exception:
                pass
            rows = conn.execute(text(sql), params).fetchall()

        results = []
        for rank, row in enumerate(rows, 1):
            results.append({
                'id': row[0],
                'repo_name': row[1],
                'repo_path': row[2],
                'branch_name': row[3],
                'commit_hash': row[4],
                'file_path': row[5],
                'file_name': row[6],
                'module_path': row[7],
                'language': row[8],
                'symbol_type': row[9],
                'symbol_name': row[10],
                'qualified_name': row[11],
                'parent_name': row[12],
                'signature': row[13],
                'docstring': row[14],
                'leading_comments': row[15],
                'code_text': row[16],
                'context_text': row[17],
                'imports_json': row[18] if row[18] else [],
                'references_json': row[19] if row[19] else [],
                'context_json': row[20] if row[20] else {},
                'line_start': row[21],
                'line_end': row[22],
                'symbol_order': row[23],
                'vector_text': row[24],
                '_similarity_score': float(row[25]),
                '_rank': rank,
            })
        return results

    def _vector_search_via_numpy(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        records, embeddings_matrix = self._load_all_vectors_cached()
        if not records:
            return []

        try:
            query_vec = np.array(query_embedding, dtype=np.float32)
            diff = embeddings_matrix - query_vec
            distances = np.einsum('ij,ij->i', diff, diff)
            n = len(distances)
            k = min(top_k, n)
            if k == n:
                top_indices = np.argsort(distances)
            else:
                cand = np.argpartition(distances, k)[:k]
                top_indices = cand[np.argsort(distances[cand])]

            results = []
            for rank, idx in enumerate(top_indices, 1):
                rec = records[idx]
                results.append({
                    **rec,
                    '_similarity_score': float(distances[idx]),
                    '_rank': rank,
                })
            return results
        except Exception as exc:
            print(f"   ❌ [book] numpy 检索失败: {exc}")
            return []

    def _candidate_key(self, item: Dict[str, Any]) -> str:
        return f"{item.get('file_path', '')}|{item.get('symbol_type', '')}|{item.get('qualified_name') or ''}"

    def _score_candidate(self, item: Dict[str, Any], query_tokens: List[str], query: str) -> Tuple[float, str]:
        score = 0.0
        reasons = []
        dist = item.get('_similarity_score')
        if dist is not None:
            vec_score = max(0.0, 1.0 - float(dist) / 2.0)
            score += vec_score * 0.7
            reasons.append(f"向量={vec_score:.2f}")

        keyword_rank = item.get('_keyword_rank')
        if isinstance(keyword_rank, int):
            score += max(0.05, 0.55 - keyword_rank * 0.02)
            reasons.append(f"全文排名={keyword_rank}")

        haystack = ' '.join([
            item.get('qualified_name') or '',
            item.get('symbol_name') or '',
            item.get('file_path') or '',
            item.get('docstring') or '',
            item.get('leading_comments') or '',
            item.get('code_text') or '',
            item.get('context_text') or '',
        ]).lower()

        hit_count = 0
        for token in query_tokens:
            token_l = token.lower()
            if token_l and token_l in haystack:
                hit_count += 1
                if token_l in (item.get('qualified_name') or '').lower():
                    score += 0.65
                elif token_l in (item.get('symbol_name') or '').lower():
                    score += 0.45
                elif token_l in (item.get('file_path') or '').lower():
                    score += 0.25
                else:
                    score += 0.15
        if hit_count:
            reasons.append(f"关键词命中={hit_count}")

        symbol_type = item.get('symbol_type')
        if symbol_type in {'function', 'method', 'class'}:
            score += 0.08
        if symbol_type == 'file':
            score += 0.03

        if query and query.lower() == (item.get('qualified_name') or '').lower():
            score += 1.2
            reasons.append('完全匹配')

        return score, '，'.join(reasons) if reasons else '综合召回'

    def _merge_candidates(self, vector_results: List[Dict[str, Any]],
                          keyword_results: List[Dict[str, Any]],
                          query: str) -> List[Dict[str, Any]]:
        query_tokens = _tokenize_query(query)
        merged: Dict[str, Dict[str, Any]] = {}
        for item in vector_results or []:
            merged[self._candidate_key(item)] = dict(item)
        for item in keyword_results or []:
            key = self._candidate_key(item)
            if key in merged:
                merged[key].update({k: v for k, v in item.items() if v is not None})
            else:
                merged[key] = dict(item)

        scored = []
        for item in merged.values():
            score, reason = self._score_candidate(item, query_tokens, query)
            item['_score'] = score
            item['_match_reason'] = reason
            scored.append(item)

        scored.sort(key=lambda r: (
            -(r.get('_score') or 0.0),
            r.get('file_path') or '',
            r.get('symbol_order') or 0,
        ))
        return scored

    def _load_file_summary(self, file_path: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            row = conn.execute(text("""
                SELECT repo_name, repo_path, branch_name, commit_hash, file_path, file_name,
                       module_path, language, symbol_type, symbol_name, qualified_name,
                       parent_name, signature, docstring, leading_comments, code_text,
                       context_text, imports_json, references_json, context_json,
                       line_start, line_end, symbol_order, vector_text, text_hash, source_hash
                FROM knowledge.book_code_knowledge
                WHERE db_name = :db_name
                  AND repo_name = :repo_name
                  AND file_path = :file_path
                  AND symbol_type = 'file'
                  AND is_active IS TRUE
                LIMIT 1
            """), {
                'db_name': self.db_name,
                'repo_name': self.repo_name,
                'file_path': file_path,
            }).fetchone()
        if not row:
            return None
        return {
            'repo_name': row[0],
            'repo_path': row[1],
            'branch_name': row[2],
            'commit_hash': row[3],
            'file_path': row[4],
            'file_name': row[5],
            'module_path': row[6],
            'language': row[7],
            'symbol_type': row[8],
            'symbol_name': row[9],
            'qualified_name': row[10],
            'parent_name': row[11],
            'signature': row[12],
            'docstring': row[13],
            'leading_comments': row[14],
            'code_text': row[15],
            'context_text': row[16],
            'imports_json': row[17] if row[17] else [],
            'references_json': row[18] if row[18] else [],
            'context_json': row[19] if row[19] else {},
            'line_start': row[20],
            'line_end': row[21],
            'symbol_order': row[22],
            'vector_text': row[23],
            'text_hash': row[24],
            'source_hash': row[25],
        }

    def search(self, query: str, top_k: int = 5, provider: Optional[str] = None,
               search_mode: Optional[str] = None,
               table_hints: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        if not self._schema_is_ready():
            return []

        mode = _normalize_book_search_mode(search_mode)
        enriched_query = _join_query_hints(query, table_hints)
        vector_results: List[Dict[str, Any]] = []
        keyword_limit = _book_full_scan_candidates() if mode == 'all' else top_k * 3

        if mode == 'all':
            print(f"📚 [book] 全量关键词检索模式: 候选 {keyword_limit}")
        elif mode == 'selected_table':
            print(f"🎯 [book] 指定表关联检索模式: {', '.join(table_hints or [])}")
        else:
            print(f"🔍 [book] 向量混合检索模式: Top {top_k}")
            from .embedding_client import get_embedding_model

            model = get_embedding_model(provider)
            query_emb = model.encode([enriched_query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)[0]
            embedding_col = self._embedding_col(provider or EMBEDDING_PROVIDER)
            if self.db_name not in self._sql_path_disabled:
                try:
                    vector_results = self._vector_search_via_holo_sql(query_emb.tolist(), top_k * 3, embedding_col=embedding_col)
                except Exception as exc:
                    with self._sql_path_lock:
                        self._sql_path_disabled.add(self.db_name)
                    print(f"   ⚠️ [book] Hologres 检索路径不可用，切换到 numpy fallback: {exc}")
                    vector_results = self._vector_search_via_numpy(query_emb.tolist(), top_k * 3)
            else:
                vector_results = self._vector_search_via_numpy(query_emb.tolist(), top_k * 3)

        keyword_results = self._keyword_search(enriched_query, keyword_limit)
        merged = self._merge_candidates(vector_results, keyword_results, enriched_query)
        if not merged:
            return []

        selected = merged[:top_k]
        file_paths = []
        for item in selected:
            file_path = item.get('file_path')
            if file_path and file_path not in file_paths:
                file_paths.append(file_path)

        for file_path in file_paths:
            summary = self._load_file_summary(file_path)
            if summary:
                selected.append({
                    **summary,
                    '_score': 0.0,
                    '_match_reason': '文件上下文补充',
                    '_expanded': True,
                })

        selected.sort(key=lambda r: (
            -(r.get('_score') or 0.0),
            r.get('_expanded', False),
            r.get('file_path') or '',
            r.get('symbol_order') or 0,
        ))

        return selected[: max(top_k, min(len(selected), top_k + len(file_paths)))]

    def _format_record_block(self, item: Dict[str, Any], file_summary: Optional[Dict[str, Any]] = None) -> str:
        file_path = item.get('file_path') or ''
        symbol_type = item.get('symbol_type') or 'file'
        symbol_name = item.get('symbol_name') or item.get('qualified_name') or item.get('file_name') or ''
        qualified_name = item.get('qualified_name') or symbol_name
        signature = item.get('signature') or ''
        docstring = _truncate_tail(item.get('docstring') or '', 400)
        leading_comments = _truncate_tail(item.get('leading_comments') or '', 300)
        code_text = _truncate_middle(item.get('context_text') or item.get('code_text') or '', _book_max_symbol_chars())
        match_reason = item.get('_match_reason') or '综合召回'
        score = item.get('_score')
        lines = [
            f"文件: {file_path}",
            f"符号: {qualified_name}",
            f"类型: {symbol_type}",
            f"签名: {signature}" if signature else '',
            f"命中原因: {match_reason}",
            f"得分: {score:.3f}" if isinstance(score, (int, float)) else '',
        ]
        if docstring:
            lines.append(f"说明: {docstring}")
        if leading_comments and leading_comments != docstring:
            lines.append(f"备注: {leading_comments}")
        if file_summary and file_summary.get('symbol_type') == 'file':
            summary_text = _truncate_middle(file_summary.get('context_text') or file_summary.get('code_text') or '', 1000)
            if summary_text:
                lines.append(f"文件上下文:\n{summary_text}")
        if code_text:
            lines.append(f"代码:\n{code_text}")
        return '\n'.join(line for line in lines if line)

    def format_results_for_prompt(self, results: Sequence[Dict[str, Any]]) -> str:
        if not results:
            return '无可用仓库代码知识'

        blocks = []
        seen_files = set()
        for item in results[:_book_prompt_items()]:
            file_path = item.get('file_path') or ''
            file_summary = None
            if file_path and file_path not in seen_files:
                file_summary = self._load_file_summary(file_path)
                seen_files.add(file_path)
            blocks.append(self._format_record_block(item, file_summary=file_summary))
        return '\n\n'.join(blocks)


class BookSyncService:
    """同步本地 Git 仓库到代码知识表。"""

    def __init__(self, db_name: Optional[str] = None, repo_path: Optional[str] = None,
                 repo_name: Optional[str] = None):
        self.db_name = db_name or os.getenv('AUTH_DB_NAME', os.getenv('APP_AUTH_DB_NAME', 'hologres'))
        self.repo_path = Path(repo_path).expanduser() if repo_path else _book_repo_path()
        self.repo_name = repo_name or _book_repo_name()
        self.engine = DatabasePoolManager.get_engine(self.db_name)
        self.store = BookKnowledgeStore(self.db_name, str(self.repo_path) if self.repo_path else None, self.repo_name, ensure_schema=True)

    def is_configured(self) -> bool:
        return self.repo_path is not None and self.repo_path.exists() and (self.repo_path / '.git').exists()

    def _fetch_all_files(self) -> List[Path]:
        if not self.is_configured():
            return []
        try:
            tracked = _git_tracked_files(self.repo_path)
            if tracked:
                return tracked
        except Exception as exc:
            print(f"   ⚠️ [book] git ls-files 失败，回退到文件遍历: {exc}")

        files = []
        for path in self.repo_path.rglob('*'):
            if path.is_file() and _should_include_file(path):
                if len(files) >= _book_max_files():
                    break
                files.append(path)
        return files

    def _collect_records_for_file(self, file_path: Path, commit_hash: str, branch_name: str) -> List[Dict[str, Any]]:
        if not _should_include_file(file_path):
            return []
        records = _build_records_for_file(
            db_name=self.store.db_name,
            repo_name=self.repo_name,
            repo_path=str(self.repo_path or ''),
            branch_name=branch_name,
            commit_hash=commit_hash,
            repo_root=self.repo_path,
            file_path=file_path,
        )
        return records

    @monitor_function
    def sync_repository(self, force: bool = False) -> Dict[str, Any]:
        if not force and not _book_sync_enabled():
            result = {
                'ok': True,
                'enabled': False,
                'repo_name': self.repo_name,
                'repo_path': str(self.repo_path) if self.repo_path else '',
                'processed_files': 0,
                'processed_records': 0,
                'stats_by_provider': {},
                'reason': 'BOOK_SYNC_ENABLED=false',
            }
            return result

        if not self.is_configured():
            raise ValueError('请配置 BOOK_REPO_PATH，且该目录必须是已 clone 的 Git 仓库')

        commit_hash = _git_commit_hash(self.repo_path)
        branch_name = _git_branch_name(self.repo_path)
        remote_url = _git_remote_url(self.repo_path)

        files = self._fetch_all_files()
        if not files:
            return {
                'ok': True,
                'enabled': True,
                'repo_name': self.repo_name,
                'repo_path': str(self.repo_path),
                'commit_hash': commit_hash,
                'branch_name': branch_name,
                'remote_url': remote_url,
                'processed_files': 0,
                'processed_records': 0,
                'stats_by_provider': {},
            }

        records: List[Dict[str, Any]] = []
        for file_path in files:
            try:
                records.extend(self._collect_records_for_file(file_path, commit_hash, branch_name))
            except Exception as exc:
                print(f"   ⚠️ [book] 处理文件失败 {file_path}: {exc}")

        stats_by_provider: Dict[str, Dict[str, int]] = {}
        for provider, model in iter_embedding_models():
            print(f"   ├─ 同步仓库代码向量（{provider}）")
            provider_stats = self.store.save_records_incrementally(model, records, [r['vector_text'] for r in records])
            stats_by_provider[provider] = provider_stats

        result = {
            'ok': True,
            'enabled': True,
            'repo_name': self.repo_name,
            'repo_path': str(self.repo_path),
            'commit_hash': commit_hash,
            'branch_name': branch_name,
            'remote_url': remote_url,
            'processed_files': len(files),
            'processed_records': len(records),
            'stats_by_provider': stats_by_provider,
        }

        global _last_sync_result
        _last_sync_result = result
        print(f"   ✅ [book] 仓库代码同步完成: {len(records)} 条知识")
        return result


def sync_book_from_source(db_name: Optional[str] = None, repo_path: Optional[str] = None,
                          repo_name: Optional[str] = None, force: bool = True) -> Dict[str, Any]:
    global _last_sync_result
    with _sync_lock:
        service = BookSyncService(db_name=db_name, repo_path=repo_path, repo_name=repo_name)
        result = service.sync_repository(force=force)
        _last_sync_result = result
        return result


def is_book_sync_monitor_running() -> bool:
    return _monitor_thread is not None and _monitor_thread.is_alive()


def get_last_book_sync_result() -> Optional[Dict[str, Any]]:
    return _last_sync_result


def start_book_sync_monitor() -> Dict[str, Any]:
    global _monitor_thread
    with _monitor_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            print("   ⚠️ 仓库代码同步线程已在运行")
            return _last_sync_result or {
                'ok': True,
                'enabled': _book_sync_enabled(),
                'repo_name': _book_repo_name(),
                'repo_path': str(_book_repo_path()) if _book_repo_path() else '',
                'processed_files': 0,
                'processed_records': 0,
                'stats_by_provider': {},
            }

        if not _book_sync_enabled():
            return _last_sync_result or {
                'ok': True,
                'enabled': False,
                'repo_name': _book_repo_name(),
                'repo_path': str(_book_repo_path()) if _book_repo_path() else '',
                'processed_files': 0,
                'processed_records': 0,
                'stats_by_provider': {},
                'scheduled': False,
            }

        def monitor_loop():
            hour = _book_sync_hour()
            minute = _book_sync_minute()
            print(f"🔁 仓库代码同步线程启动，每天 {hour:02d}:{minute:02d} 执行一次")
            while True:
                try:
                    sync_book_from_source(force=False)
                except Exception as exc:
                    print(f"   ❌ 仓库代码同步失败: {exc}")

                now = datetime.now()
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if now >= next_run:
                    next_run += timedelta(days=1)
                sleep_seconds = max(60, (next_run - now).total_seconds())
                print(f"   ├─ 距下次同步还有 {sleep_seconds / 3600:.1f} 小时 ({next_run.strftime('%Y-%m-%d %H:%M')})")
                time.sleep(sleep_seconds)

        _monitor_thread = threading.Thread(target=monitor_loop, daemon=True, name='BookCodeSyncMonitor')
        _monitor_thread.start()
        print("   ✅ 仓库代码同步线程已启动")
        return _last_sync_result or {
            'ok': True,
            'enabled': True,
            'repo_name': _book_repo_name(),
            'repo_path': str(_book_repo_path()) if _book_repo_path() else '',
            'processed_files': 0,
            'processed_records': 0,
            'stats_by_provider': {},
            'scheduled': True,
        }


if __name__ == '__main__':
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description='仓库代码知识同步工具')
    parser.add_argument('command', nargs='?', default='sync', choices=['sync', 'monitor'])
    parser.add_argument('--db-name', default=None)
    parser.add_argument('--repo-path', default=None)
    parser.add_argument('--repo-name', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.command == 'monitor':
        start_book_sync_monitor()
        while True:
            time.sleep(3600)
    else:
        result = sync_book_from_source(args.db_name, args.repo_path, args.repo_name, force=args.force)
        print(_json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))

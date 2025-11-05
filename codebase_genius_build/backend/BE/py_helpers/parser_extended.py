import ast
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, TypedDict


class ParseResult(TypedDict):
    language: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    imports: List[str]
    calls_to_other_files: List[str]


class PythonASTVisitor(ast.NodeVisitor):
    """
    AST visitor to extract nodes, edges, and imports from Python source with context.
    Tracks current scope for better edge attribution.
    """

    def __init__(self) -> None:
        self.result: ParseResult = {
            'language': 'python',
            'nodes': [],
            'edges': [],
            'imports': [],
            'calls_to_other_files': []
        }
        self.current_scope: Optional[str] = None  # e.g., 'fn:func_name' or 'class:ClassName'
        self.imported_modules: set[str] = set()  # Track imported names for external calls

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        node_id = f"fn:{node.name}:{node.lineno}"
        self.result['nodes'].append({
            'id': node_id,
            'type': 'function',
            'name': node.name,
            'lineno': node.lineno
        })
        self.current_scope = node_id
        self.generic_visit(node)
        self.current_scope = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        node_id = f"class:{node.name}:{node.lineno}"
        self.result['nodes'].append({
            'id': node_id,
            'type': 'class',
            'name': node.name,
            'lineno': node.lineno
        })
        self.current_scope = node_id
        self.generic_visit(node)
        self.current_scope = None

    def visit_Call(self, node: ast.Call) -> None:
        if self.current_scope:
            func = node.func
            target_name = self._extract_call_target(func)
            if target_name:
                self.result['edges'].append({
                    'from': self.current_scope,
                    'to': target_name,
                    'type': 'call'
                })
                # Detect potential external call
                if self._is_external_call(func, target_name):
                    self.result['calls_to_other_files'].append(target_name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            mod_name = alias.name.split('.')[0]  # Top-level module
            self.result['imports'].append(alias.name)
            self.imported_modules.add(alias.asname or alias.name.split('.')[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.result['imports'].append(node.module)
            self.imported_modules.add(node.module.split('.')[0])
        for alias in node.names:
            if alias.name == '*':
                continue  # Skip wildcard for simplicity
            import_name = f"{node.module}.{alias.name}" if node.module else alias.name
            self.result['imports'].append(import_name)
            self.imported_modules.add(alias.asname or alias.name)

    def _extract_call_target(self, func: ast.expr) -> Optional[str]:
        """Extract the target name or qualified name from a Call's func."""
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            # Recursively build qualified name
            base = self._extract_call_target(func.value)
            if base:
                return f"{base}.{func.attr}"
            return func.attr
        return None

    def _is_external_call(self, func: ast.expr, target: str) -> bool:
        """Check if call is likely to an external module (imported)."""
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return func.value.id in self.imported_modules
        # For simple cases like imported_func()
        if isinstance(func, ast.Name) and func.id in self.imported_modules:
            return True
        return False


def parse_python_file(path: str | Path) -> ParseResult:
    """
    Parse a Python file to extract structural elements: functions/classes (nodes),
    calls (edges), imports, and potential external calls.

    Args:
        path (str | Path): Path to the Python file.

    Returns:
        ParseResult: Dictionary with extracted info.
    """
    path = Path(path).resolve()
    if not path.is_file():
        return {
            'language': 'python',
            'nodes': [],
            'edges': [],
            'imports': [],
            'calls_to_other_files': []
        }

    try:
        with path.open('r', encoding='utf-8') as f:
            src = f.read()
    except (IOError, UnicodeDecodeError):
        return {
            'language': 'python',
            'nodes': [],
            'edges': [],
            'imports': [],
            'calls_to_other_files': []
        }

    try:
        tree = ast.parse(src, filename=str(path))
        visitor = PythonASTVisitor()
        visitor.visit(tree)
        return visitor.result
    except SyntaxError:
        return {
            'language': 'python',
            'nodes': [],
            'edges': [],
            'imports': [],
            'calls_to_other_files': []
        }


def parse_jac_file(path: str | Path) -> ParseResult:
    """
    Parse a Jac file using regex to extract nodes, walkers, imports, and calls.
    Improved regex for accuracy; adds approximate line numbers.

    Args:
        path (str | Path): Path to the Jac file.

    Returns:
        ParseResult: Dictionary with extracted info.
    """
    path = Path(path).resolve()
    if not path.is_file():
        return {
            'language': 'jac',
            'nodes': [],
            'edges': [],
            'imports': [],
            'calls_to_other_files': []
        }

    try:
        with path.open('r', encoding='utf-8') as f:
            lines = f.readlines()
        src = ''.join(lines)
    except (IOError, UnicodeDecodeError):
        return {
            'language': 'jac',
            'nodes': [],
            'edges': [],
            'imports': [],
            'calls_to_other_files': []
        }

    # Improved regex patterns based on Jac syntax
    node_pattern = re.compile(r'^\s*node\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:{|$)', re.MULTILINE)
    walker_pattern = re.compile(r'^\s*walker\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:{|$)', re.MULTILINE)
    import_pattern = re.compile(r'^\s*import\s+([A-Za-z0-9_][A-Za-z0-9_.]*)\s*(?:from|import|$)', re.MULTILINE)
    call_pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*::\s*([A-Za-z_][A-Za-z0-9_]*)', re.DOTALL)

    result: ParseResult = {
        'language': 'jac',
        'nodes': [],
        'edges': [],
        'imports': [],
        'calls_to_other_files': []
    }

    # Extract nodes with line numbers
    for match in node_pattern.finditer(src):
        name = match.group(1)
        lineno = src[:match.start()].count('\n') + 1
        result['nodes'].append({
            'id': f'jac_node:{name}:{lineno}',
            'type': 'node',
            'name': name,
            'lineno': lineno
        })

    # Extract walkers with line numbers
    for match in walker_pattern.finditer(src):
        name = match.group(1)
        lineno = src[:match.start()].count('\n') + 1
        result['nodes'].append({
            'id': f'jac_walker:{name}:{lineno}',
            'type': 'walker',
            'name': name,
            'lineno': lineno
        })

    # Extract imports
    for match in import_pattern.finditer(src):
        result['imports'].append(match.group(1))

    # Extract calls (assume external if :: used)
    for match in call_pattern.finditer(src):
        module, func = match.groups()
        target = f"{module}::{func}"
        result['edges'].append({
            'from': None,  # TODO: Scope tracking would require full parser
            'to': target,
            'type': 'call'
        })
        result['calls_to_other_files'].append(target)

    return result


def parse_file(path: str | Path) -> ParseResult:
    """
    Dispatch parser based on file extension.

    Args:
        path (str | Path): Path to the file.

    Returns:
        ParseResult: Extracted info or empty for unknown types.
    """
    path = Path(path)
    if path.suffix == '.py':
        return parse_python_file(path)
    elif path.suffix == '.jac':
        return parse_jac_file(path)
    else:
        return {
            'language': 'unknown',
            'nodes': [],
            'edges': [],
            'imports': [],
            'calls_to_other_files': []
        }
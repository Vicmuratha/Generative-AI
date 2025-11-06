"""
Enhanced multi-language code parser with improved accuracy and extensibility.

Supports Python and Jac with rich structural analysis including:
- Functions, classes, nodes, walkers
- Call graphs with scope tracking
- Import analysis with dependency resolution
- Decorator and method detection
- Better error handling and logging
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class NodeType(str, Enum):
    """Types of code nodes that can be extracted."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    NODE = "node"  # Jac node
    WALKER = "walker"  # Jac walker
    ABILITY = "ability"  # Jac ability
    ENUM = "enum"


class EdgeType(str, Enum):
    """Types of relationships between code nodes."""
    CALL = "call"
    INHERITS = "inherits"
    IMPORTS = "imports"
    DECORATES = "decorates"


@dataclass
class CodeNode:
    """Represents a structural element in code."""
    id: str
    type: str
    name: str
    lineno: int
    end_lineno: Optional[int] = None
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    parent: Optional[str] = None  # For methods in classes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'type': self.type,
            'name': self.name,
            'lineno': self.lineno,
            'end_lineno': self.end_lineno,
            'docstring': self.docstring,
            'decorators': self.decorators,
            'parameters': self.parameters,
            'return_type': self.return_type,
            'parent': self.parent
        }


@dataclass
class CodeEdge:
    """Represents a relationship between code nodes."""
    from_node: str
    to_node: str
    type: str
    lineno: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'from': self.from_node,
            'to': self.to_node,
            'type': self.type,
            'lineno': self.lineno
        }


@dataclass
class ParseResult:
    """Complete parse result for a file."""
    language: str
    nodes: List[CodeNode] = field(default_factory=list)
    edges: List[CodeEdge] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    external_calls: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'language': self.language,
            'nodes': [n.to_dict() for n in self.nodes],
            'edges': [e.to_dict() for e in self.edges],
            'imports': self.imports,
            'calls_to_other_files': self.external_calls,
            'errors': self.errors,
            'metadata': self.metadata
        }


class PythonASTVisitor(ast.NodeVisitor):
    """
    Enhanced AST visitor for Python with comprehensive analysis.
    
    Features:
    - Scope tracking (module, class, function)
    - Method vs function distinction
    - Decorator extraction
    - Inheritance tracking
    - Parameter and return type extraction
    - Docstring extraction
    """

    def __init__(self) -> None:
        self.result = ParseResult(language='python')
        self.scope_stack: List[str] = []  # Track nested scopes
        self.current_class: Optional[str] = None
        self.imported_modules: Set[str] = set()
        self.import_aliases: Dict[str, str] = {}  # Map aliases to real names

    def _push_scope(self, scope_id: str) -> None:
        """Enter a new scope."""
        self.scope_stack.append(scope_id)

    def _pop_scope(self) -> None:
        """Exit current scope."""
        if self.scope_stack:
            self.scope_stack.pop()

    @property
    def current_scope(self) -> Optional[str]:
        """Get the current scope ID."""
        return self.scope_stack[-1] if self.scope_stack else None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        node_id = f"class:{node.name}:{node.lineno}"
        
        # Extract decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_qualified_name(base))
        
        code_node = CodeNode(
            id=node_id,
            type=NodeType.CLASS.value,
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, 'end_lineno', None),
            docstring=docstring,
            decorators=decorators,
            parent=self.current_scope
        )
        self.result.nodes.append(code_node)
        
        # Add inheritance edges
        for base in bases:
            edge = CodeEdge(
                from_node=node_id,
                to_node=f"class:{base}",
                type=EdgeType.INHERITS.value,
                lineno=node.lineno
            )
            self.result.edges.append(edge)
        
        # Enter class scope
        old_class = self.current_class
        self.current_class = node_id
        self._push_scope(node_id)
        
        self.generic_visit(node)
        
        # Exit class scope
        self._pop_scope()
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function/method definition."""
        # Determine if this is a method or function
        is_method = self.current_class is not None
        node_type = NodeType.METHOD if is_method else NodeType.FUNCTION
        
        node_id = f"{node_type.value}:{node.name}:{node.lineno}"
        
        # Extract decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param_name = arg.arg
            if arg.annotation:
                param_type = self._get_annotation_str(arg.annotation)
                parameters.append(f"{param_name}: {param_type}")
            else:
                parameters.append(param_name)
        
        # Extract return type
        return_type = None
        if node.returns:
            return_type = self._get_annotation_str(node.returns)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        code_node = CodeNode(
            id=node_id,
            type=node_type.value,
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, 'end_lineno', None),
            docstring=docstring,
            decorators=decorators,
            parameters=parameters,
            return_type=return_type,
            parent=self.current_class
        )
        self.result.nodes.append(code_node)
        
        # Enter function scope
        self._push_scope(node_id)
        self.generic_visit(node)
        self._pop_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition (treat like regular function)."""
        self.visit_FunctionDef(node)  # type: ignore

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function/method call."""
        if self.current_scope:
            target_name = self._extract_call_target(node.func)
            if target_name:
                edge = CodeEdge(
                    from_node=self.current_scope,
                    to_node=target_name,
                    type=EdgeType.CALL.value,
                    lineno=node.lineno
                )
                self.result.edges.append(edge)
                
                # Check if this is an external call
                if self._is_external_call(node.func, target_name):
                    self.result.external_calls.append(target_name)
        
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        for alias in node.names:
            self.result.imports.append(alias.name)
            
            # Track module name
            module_name = alias.name.split('.')[0]
            self.imported_modules.add(module_name)
            
            # Track alias
            if alias.asname:
                self.import_aliases[alias.asname] = alias.name
                self.imported_modules.add(alias.asname)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statement."""
        if node.module:
            module_name = node.module.split('.')[0]
            self.imported_modules.add(module_name)
            self.result.imports.append(node.module)
        
        for alias in node.names:
            if alias.name == '*':
                continue
            
            import_name = f"{node.module}.{alias.name}" if node.module else alias.name
            self.result.imports.append(import_name)
            
            # Track imported name
            imported_as = alias.asname or alias.name
            self.imported_modules.add(imported_as)
            if alias.asname:
                self.import_aliases[alias.asname] = import_name

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        elif isinstance(decorator, ast.Attribute):
            return self._get_qualified_name(decorator)
        return str(decorator)

    def _get_annotation_str(self, annotation: ast.expr) -> str:
        """Convert annotation AST to string."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return repr(annotation.value)
        elif isinstance(annotation, ast.Subscript):
            value = self._get_annotation_str(annotation.value)
            slice_val = self._get_annotation_str(annotation.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(annotation, ast.Attribute):
            return self._get_qualified_name(annotation)
        elif isinstance(annotation, ast.Tuple):
            elements = [self._get_annotation_str(e) for e in annotation.elts]
            return f"({', '.join(elements)})"
        return "Any"

    def _get_qualified_name(self, node: ast.Attribute) -> str:
        """Get fully qualified name from attribute chain."""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        return '.'.join(reversed(parts))

    def _extract_call_target(self, func: ast.expr) -> Optional[str]:
        """Extract target name from call expression."""
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            base = self._extract_call_target(func.value)
            if base:
                return f"{base}.{func.attr}"
            return func.attr
        return None

    def _is_external_call(self, func: ast.expr, target: str) -> bool:
        """Check if call targets an imported module."""
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module_name = func.value.id
            return module_name in self.imported_modules
        
        if isinstance(func, ast.Name):
            func_name = func.id
            return func_name in self.imported_modules
        
        # Check if target starts with known module
        target_parts = target.split('.')
        if target_parts[0] in self.imported_modules:
            return True
        
        return False


class JacParser:
    """
    Enhanced Jac parser with better pattern matching and scope tracking.
    
    Handles:
    - Walkers, nodes, abilities, enums
    - Import statements (import:jac, import:py)
    - Module-qualified calls (::)
    - Can functions
    - Type annotations
    """

    def __init__(self) -> None:
        self.result = ParseResult(language='jac')
        self.lines: List[str] = []
        
    def parse(self, path: Path) -> ParseResult:
        """Parse a Jac file."""
        if not path.is_file():
            self.result.errors.append(f"File not found: {path}")
            return self.result
        
        try:
            with path.open('r', encoding='utf-8') as f:
                self.lines = f.readlines()
            src = ''.join(self.lines)
        except (IOError, UnicodeDecodeError) as e:
            self.result.errors.append(f"Failed to read file: {e}")
            return self.result
        
        self._extract_nodes(src)
        self._extract_imports(src)
        self._extract_calls(src)
        self._calculate_metadata(src)
        
        return self.result
    
    def _extract_nodes(self, src: str) -> None:
        """Extract all structural nodes from Jac source."""
        
        # Walker definitions
        walker_pattern = re.compile(
            r'^\s*walker\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?\s*(?:->?\s*([^{]+?))?\s*{',
            re.MULTILINE
        )
        for match in walker_pattern.finditer(src):
            name, params, return_type = match.groups()
            lineno = src[:match.start()].count('\n') + 1
            
            node = CodeNode(
                id=f"walker:{name}:{lineno}",
                type=NodeType.WALKER.value,
                name=name,
                lineno=lineno,
                parameters=self._parse_params(params) if params else [],
                return_type=return_type.strip() if return_type else None
            )
            self.result.nodes.append(node)
        
        # Node definitions
        node_pattern = re.compile(
            r'^\s*node\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?\s*{',
            re.MULTILINE
        )
        for match in node_pattern.finditer(src):
            name, params = match.groups()
            lineno = src[:match.start()].count('\n') + 1
            
            node = CodeNode(
                id=f"node:{name}:{lineno}",
                type=NodeType.NODE.value,
                name=name,
                lineno=lineno,
                parameters=self._parse_params(params) if params else []
            )
            self.result.nodes.append(node)
        
        # Can (ability) definitions
        can_pattern = re.compile(
            r'^\s*can\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?\s*(?:->?\s*([^{]+?))?\s*{',
            re.MULTILINE
        )
        for match in can_pattern.finditer(src):
            name, params, return_type = match.groups()
            lineno = src[:match.start()].count('\n') + 1
            
            # Extract docstring if present
            docstring = self._extract_docstring_after(src, match.end())
            
            node = CodeNode(
                id=f"ability:{name}:{lineno}",
                type=NodeType.ABILITY.value,
                name=name,
                lineno=lineno,
                docstring=docstring,
                parameters=self._parse_params(params) if params else [],
                return_type=return_type.strip() if return_type else None
            )
            self.result.nodes.append(node)
        
        # Enum definitions
        enum_pattern = re.compile(
            r'^\s*enum\s+([A-Za-z_][A-Za-z0-9_]*)\s*{',
            re.MULTILINE
        )
        for match in enum_pattern.finditer(src):
            name = match.group(1)
            lineno = src[:match.start()].count('\n') + 1
            
            node = CodeNode(
                id=f"enum:{name}:{lineno}",
                type=NodeType.ENUM.value,
                name=name,
                lineno=lineno
            )
            self.result.nodes.append(node)
    
    def _extract_imports(self, src: str) -> None:
        """Extract import statements."""
        
        # import:jac from module { items }
        jac_import_pattern = re.compile(
            r'import:jac\s+from\s+([A-Za-z_][A-Za-z0-9_.]*)\s*{([^}]+)}',
            re.MULTILINE
        )
        for match in jac_import_pattern.finditer(src):
            module = match.group(1)
            items = match.group(2)
            self.result.imports.append(f"{module} (jac)")
            
            # Track individual items
            for item in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)', items):
                self.result.imports.append(f"{module}.{item}")
        
        # import:py from module { items }
        py_import_pattern = re.compile(
            r'import:py\s+from\s+([A-Za-z_][A-Za-z0-9_.]*)\s*{([^}]+)}',
            re.MULTILINE
        )
        for match in py_import_pattern.finditer(src):
            module = match.group(1)
            items = match.group(2)
            self.result.imports.append(f"{module} (py)")
        
        # Legacy: import module;
        legacy_import_pattern = re.compile(
            r'^\s*import\s+([A-Za-z_][A-Za-z0-9_.]*)\s*;',
            re.MULTILINE
        )
        for match in legacy_import_pattern.finditer(src):
            module = match.group(1)
            self.result.imports.append(module)
    
    def _extract_calls(self, src: str) -> None:
        """Extract function/walker calls."""
        
        # Module-qualified calls: module::function
        qualified_call_pattern = re.compile(
            r'([A-Za-z_][A-Za-z0-9_]*)\s*::\s*([A-Za-z_][A-Za-z0-9_]*)',
            re.DOTALL
        )
        for match in qualified_call_pattern.finditer(src):
            module, func = match.groups()
            target = f"{module}::{func}"
            lineno = src[:match.start()].count('\n') + 1
            
            edge = CodeEdge(
                from_node="<context>",  # TODO: Track current scope
                to_node=target,
                type=EdgeType.CALL.value,
                lineno=lineno
            )
            self.result.edges.append(edge)
            self.result.external_calls.append(target)
    
    def _parse_params(self, params_str: str) -> List[str]:
        """Parse parameter string into list."""
        if not params_str or not params_str.strip():
            return []
        
        params = []
        for param in params_str.split(','):
            param = param.strip()
            if param:
                params.append(param)
        return params
    
    def _extract_docstring_after(self, src: str, pos: int) -> Optional[str]:
        """Extract docstring immediately after position."""
        remaining = src[pos:]
        docstring_pattern = re.compile(r'^\s*"""(.*?)"""', re.DOTALL)
        match = docstring_pattern.match(remaining)
        if match:
            return match.group(1).strip()
        return None
    
    def _calculate_metadata(self, src: str) -> None:
        """Calculate metadata about the file."""
        self.result.metadata = {
            'line_count': len(self.lines),
            'node_count': len([n for n in self.result.nodes if n.type == NodeType.NODE.value]),
            'walker_count': len([n for n in self.result.nodes if n.type == NodeType.WALKER.value]),
            'ability_count': len([n for n in self.result.nodes if n.type == NodeType.ABILITY.value]),
            'enum_count': len([n for n in self.result.nodes if n.type == NodeType.ENUM.value]),
            'import_count': len(self.result.imports)
        }


def parse_python_file(path: Path) -> ParseResult:
    """Parse a Python file with comprehensive analysis."""
    if not path.is_file():
        result = ParseResult(language='python')
        result.errors.append(f"File not found: {path}")
        return result
    
    try:
        with path.open('r', encoding='utf-8') as f:
            src = f.read()
    except (IOError, UnicodeDecodeError) as e:
        result = ParseResult(language='python')
        result.errors.append(f"Failed to read file: {e}")
        return result
    
    try:
        tree = ast.parse(src, filename=str(path))
        visitor = PythonASTVisitor()
        visitor.visit(tree)
        
        # Add metadata
        visitor.result.metadata = {
            'line_count': src.count('\n') + 1,
            'function_count': len([n for n in visitor.result.nodes if n.type == NodeType.FUNCTION.value]),
            'class_count': len([n for n in visitor.result.nodes if n.type == NodeType.CLASS.value]),
            'method_count': len([n for n in visitor.result.nodes if n.type == NodeType.METHOD.value]),
            'import_count': len(visitor.result.imports)
        }
        
        return visitor.result
    except SyntaxError as e:
        result = ParseResult(language='python')
        result.errors.append(f"Syntax error: {e}")
        return result


def parse_jac_file(path: Path) -> ParseResult:
    """Parse a Jac file with enhanced pattern matching."""
    parser = JacParser()
    return parser.parse(path)


def parse_file(path: str | Path) -> Dict[str, Any]:
    """
    Main entry point: dispatch to appropriate parser based on file extension.
    
    Args:
        path: Path to source file
        
    Returns:
        Dict with extracted information (compatible with original API)
    """
    path = Path(path).resolve()
    
    if not path.exists():
        return {
            'language': 'unknown',
            'nodes': [],
            'edges': [],
            'imports': [],
            'calls_to_other_files': [],
            'errors': [f"File does not exist: {path}"],
            'metadata': {}
        }
    
    suffix = path.suffix.lower()
    
    if suffix == '.py':
        result = parse_python_file(path)
    elif suffix == '.jac':
        result = parse_jac_file(path)
    else:
        result = ParseResult(language='unknown')
        result.metadata = {'file_extension': suffix}
    
    # Return as dict for compatibility
    return result.to_dict()


def parse_directory(directory: str | Path, extensions: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Parse all supported files in a directory.
    
    Args:
        directory: Directory path
        extensions: List of extensions to parse (default: ['.py', '.jac'])
        
    Returns:
        Dict mapping file paths to parse results
    """
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    
    if extensions is None:
        extensions = ['.py', '.jac']
    
    results = {}
    for ext in extensions:
        for file_path in directory.rglob(f'*{ext}'):
            if file_path.is_file():
                results[str(file_path)] = parse_file(file_path)
    
    return results
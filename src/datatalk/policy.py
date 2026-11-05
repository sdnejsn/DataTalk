"""Best-effort static restrictions; this module is not a security sandbox."""

from __future__ import annotations

import ast

ALLOWED_MODULES = {
    "pandas",
    "numpy",
    "matplotlib.pyplot",
    "math",
    "statistics",
    "datetime",
    "collections",
    "itertools",
    "functools",
}
BLOCKED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "help",
    "breakpoint",
    "exit",
    "quit",
    "memoryview",
    "type",
    "object",
    "classmethod",
    "staticmethod",
    "property",
}
BLOCKED_ATTRIBUTES = {
    "load",
    "loads",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
    "loadtxt",
    "genfromtxt",
    "fromfile",
    "tofile",
    "memmap",
    "ctypeslib",
    "ctypes",
    "lib",
    "io",
    "compat",
    "testing",
    "show",
    "imsave",
    "imread",
    "get_data",
    "eval",
    "query",
    "system",
    "popen",
    "subprocess",
    "os",
    "sys",
    "builtins",
    "importlib",
}
ALLOWED_CONVERSIONS = {
    "to_datetime",
    "to_numeric",
    "to_timedelta",
    "to_string",
    "to_dict",
    "to_list",
    "to_numpy",
    "to_period",
    "to_timestamp",
    "to_frame",
    "to_records",
    "to_pydatetime",
    "to_flat_index",
}


def is_prohibited_attribute(name: str) -> bool:
    return (
        name.startswith("_")
        or name in BLOCKED_ATTRIBUTES
        or name.startswith("read_")
        or (name.startswith("to_") and name not in ALLOWED_CONVERSIONS)
    )


def validate_code(code: str) -> None:
    if not code.strip() or len(code) > 30_000:
        raise ValueError("Code must be nonempty and at most 30000 characters")
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name not in ALLOWED_MODULES for alias in node.names):
                raise ValueError("Import is outside the permitted analysis modules")
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_MODULES:
                raise ValueError("Import is outside the permitted analysis modules")
            if any(
                is_prohibited_attribute(alias.name) or alias.name == "*" for alias in node.names
            ):
                raise ValueError("Private, file-access and wildcard imports are prohibited")
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in BLOCKED_NAMES:
                raise ValueError(f"Name is prohibited: {node.id}")
        elif isinstance(node, ast.Attribute):
            name = node.attr
            if is_prohibited_attribute(name):
                raise ValueError(f"Attribute is prohibited: {name}")
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await)):
            raise ValueError("Classes and asynchronous code are not supported")

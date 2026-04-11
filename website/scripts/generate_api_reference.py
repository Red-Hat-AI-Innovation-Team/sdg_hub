#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract API documentation from the sdg_hub package and generate JSON for the docs site.

Usage:
    cd /path/to/repo-root
    uv run python website/scripts/generate_api_reference.py

Output:
    website/public/api-reference.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import ast
import importlib
import inspect
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # website/scripts/../../
SRC_DIR = REPO_ROOT / "src"
OUTPUT_PATH = REPO_ROOT / "website" / "public" / "api-reference.json"

# Ensure `src/` is importable
sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Module map: category -> subcategory -> list of (module_path, [class_names])
# ---------------------------------------------------------------------------
MODULE_MAP: dict[str, dict[str, list[tuple[str, list[str] | None]]]] = {
    "blocks": {
        "base": [
            ("sdg_hub.core.blocks.base", ["BaseBlock"]),
        ],
        "registry": [
            ("sdg_hub.core.blocks.registry", ["BlockRegistry", "BlockMetadata"]),
        ],
        "llm": [
            ("sdg_hub.core.blocks.llm.llm_chat_block", ["LLMChatBlock"]),
            (
                "sdg_hub.core.blocks.llm.prompt_builder_block",
                [
                    "PromptBuilderBlock",
                    "ChatMessage",
                    "MessageTemplate",
                    "PromptTemplateConfig",
                    "PromptRenderer",
                ],
            ),
            (
                "sdg_hub.core.blocks.llm.llm_response_extractor_block",
                ["LLMResponseExtractorBlock"],
            ),
        ],
        "parsing": [
            ("sdg_hub.core.blocks.parsing.json_parser_block", None),
            ("sdg_hub.core.blocks.parsing.regex_parser_block", None),
            ("sdg_hub.core.blocks.parsing.tag_parser_block", None),
            ("sdg_hub.core.blocks.parsing.text_parser_block", None),
        ],
        "transform": [
            ("sdg_hub.core.blocks.transform.duplicate_columns", None),
            ("sdg_hub.core.blocks.transform.index_based_mapper", None),
            ("sdg_hub.core.blocks.transform.json_structure_block", None),
            ("sdg_hub.core.blocks.transform.melt_columns", None),
            ("sdg_hub.core.blocks.transform.rename_columns", None),
            ("sdg_hub.core.blocks.transform.row_multiplier", None),
            ("sdg_hub.core.blocks.transform.sampler", None),
            ("sdg_hub.core.blocks.transform.text_concat", None),
            ("sdg_hub.core.blocks.transform.uniform_col_val_setter", None),
        ],
        "filtering": [
            ("sdg_hub.core.blocks.filtering.column_value_filter", None),
        ],
        "agent": [
            ("sdg_hub.core.blocks.agent.agent_block", None),
            (
                "sdg_hub.core.blocks.agent.agent_response_extractor_block",
                None,
            ),
        ],
        "mcp": [
            ("sdg_hub.core.blocks.mcp.mcp_agent_block", None),
        ],
    },
    "flow": {
        "base": [
            ("sdg_hub.core.flow.base", ["Flow"]),
        ],
        "registry": [
            ("sdg_hub.core.flow.registry", ["FlowRegistry", "FlowRegistryEntry"]),
        ],
        "metadata": [
            (
                "sdg_hub.core.flow.metadata",
                [
                    "FlowMetadata",
                    "RecommendedModels",
                    "DatasetRequirements",
                    "ModelOption",
                    "ModelCompatibility",
                ],
            ),
        ],
    },
    "connectors": {
        "base": [
            (
                "sdg_hub.core.connectors.base",
                ["BaseConnector", "ConnectorConfig"],
            ),
        ],
        "registry": [
            ("sdg_hub.core.connectors.registry", ["ConnectorRegistry"]),
        ],
        "agent": [
            ("sdg_hub.core.connectors.agent.base", ["BaseAgentConnector"]),
            ("sdg_hub.core.connectors.agent.langflow", ["LangflowConnector"]),
            ("sdg_hub.core.connectors.agent.langgraph", ["LangGraphConnector"]),
        ],
    },
}


# ---------------------------------------------------------------------------
# AST helpers -- extract decorator arguments that aren't available at runtime
# ---------------------------------------------------------------------------


def _extract_decorator_info(source_file: str, class_name: str) -> dict[str, Any] | None:
    """Parse source file with ast to find decorator arguments for *class_name*."""
    try:
        source_path = Path(source_file)
        if not source_path.exists():
            return None
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=source_file)
    except Exception:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for decorator in node.decorator_list:
            info = _parse_decorator_node(decorator)
            if info:
                return info
    return None


def _ast_value(node: ast.expr) -> Any:
    """Convert an AST literal node to a Python value."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_ast_value(node.value)}.{node.attr}"
    if isinstance(node, ast.List):
        return [_ast_value(e) for e in node.elts]
    return repr(ast.dump(node))


def _parse_decorator_node(node: ast.expr) -> dict[str, Any] | None:
    """Extract name and arguments from a decorator node."""
    # @Foo.bar(...)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            name = f"{_ast_value(func.value)}.{func.attr}"
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            return None
        args = [_ast_value(a) for a in node.args]
        kwargs = {kw.arg: _ast_value(kw.value) for kw in node.keywords if kw.arg}
        return {"decorator": name, "args": args, "kwargs": kwargs}
    # @Foo.bar (no call)
    if isinstance(node, ast.Attribute):
        return {
            "decorator": f"{_ast_value(node.value)}.{node.attr}",
            "args": [],
            "kwargs": {},
        }
    if isinstance(node, ast.Name):
        return {"decorator": node.id, "args": [], "kwargs": {}}
    return None


# ---------------------------------------------------------------------------
# Type annotation helpers
# ---------------------------------------------------------------------------


def _annotation_to_str(annotation: Any) -> str:
    """Convert a type annotation to a readable string."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return ""
    if isinstance(annotation, str):
        return annotation
    if hasattr(annotation, "__module__") and hasattr(annotation, "__qualname__"):
        # Use short name for builtins and well-known types
        module = getattr(annotation, "__module__", "")
        name = getattr(annotation, "__qualname__", str(annotation))
        if module == "builtins":
            return name
        return f"{module}.{name}" if module else name
    return str(annotation)


def _format_type(annotation: Any) -> str:
    """Format type annotation, handling typing generics."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return ""
    try:
        origin = getattr(annotation, "__origin__", None)
        args = getattr(annotation, "__args__", None)
        if origin is not None:
            origin_name = getattr(origin, "__name__", str(origin))
            if args:
                args_str = ", ".join(_format_type(a) for a in args)
                return f"{origin_name}[{args_str}]"
            return origin_name
        return _annotation_to_str(annotation)
    except Exception:
        return str(annotation)


# ---------------------------------------------------------------------------
# Pydantic field extraction
# ---------------------------------------------------------------------------


def _is_pydantic_model(cls: type) -> bool:
    """Check if cls is a Pydantic BaseModel subclass."""
    try:
        from pydantic import BaseModel as PydanticBase

        return isinstance(cls, type) and issubclass(cls, PydanticBase)
    except ImportError:
        return False


def _extract_pydantic_fields(cls: type) -> list[dict[str, Any]]:
    """Extract Pydantic model_fields from a class."""
    fields: list[dict[str, Any]] = []
    model_fields = getattr(cls, "model_fields", {})
    for name, field_info in model_fields.items():
        # Skip private / internal fields
        if name.startswith("_"):
            continue
        field_data: dict[str, Any] = {"name": name}

        # Type
        annotation = field_info.annotation
        field_data["type"] = _format_type(annotation) if annotation else ""

        # Required
        field_data["required"] = field_info.is_required()

        # Default
        default = field_info.default
        if default is not None and not callable(default):
            # PydanticUndefined is the sentinel for required fields
            sentinel_cls = type(default)
            if sentinel_cls.__name__ == "PydanticUndefinedType":
                field_data["default"] = None
            else:
                try:
                    json.dumps(default)  # ensure serializable
                    field_data["default"] = default
                except (TypeError, ValueError):
                    field_data["default"] = repr(default)
        else:
            field_data["default"] = None

        # Default factory
        if field_info.default_factory is not None:
            try:
                factory_val = field_info.default_factory()
                json.dumps(factory_val)
                field_data["default"] = factory_val
            except Exception:
                field_data["default"] = "factory()"

        # Description
        field_data["description"] = field_info.description or ""

        # Exclude flag
        exclude_val = getattr(field_info, "exclude", None)
        if exclude_val:
            field_data["exclude"] = True

        fields.append(field_data)
    return fields


# ---------------------------------------------------------------------------
# Dataclass field extraction
# ---------------------------------------------------------------------------


def _is_dataclass(cls: type) -> bool:
    """Check if cls is a dataclass."""
    import dataclasses

    return dataclasses.is_dataclass(cls) and isinstance(cls, type)


def _extract_dataclass_fields(cls: type) -> list[dict[str, Any]]:
    """Extract fields from a dataclass."""
    import dataclasses

    fields: list[dict[str, Any]] = []
    for f in dataclasses.fields(cls):
        if f.name.startswith("_"):
            continue
        field_data: dict[str, Any] = {
            "name": f.name,
            "type": _format_type(f.type) if f.type else "",
            "required": f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING,
            "description": "",
        }
        if f.default is not dataclasses.MISSING:
            try:
                json.dumps(f.default)
                field_data["default"] = f.default
            except (TypeError, ValueError):
                field_data["default"] = repr(f.default)
        elif f.default_factory is not dataclasses.MISSING:
            try:
                val = f.default_factory()
                json.dumps(val)
                field_data["default"] = val
            except Exception:
                field_data["default"] = "factory()"
        else:
            field_data["default"] = None
        fields.append(field_data)
    return fields


# ---------------------------------------------------------------------------
# Method extraction
# ---------------------------------------------------------------------------


def _is_public_method(name: str) -> bool:
    """Return True for public methods (not _private, not __dunder__ except __init__)."""
    if name == "__init__":
        return True
    if name.startswith("_"):
        return False
    return True


def _extract_methods(cls: type) -> list[dict[str, Any]]:
    """Extract public methods from a class."""
    methods: list[dict[str, Any]] = []
    # Get methods defined directly on this class (not inherited), plus important
    # inherited abstract methods
    own_attrs = set(cls.__dict__.keys())

    for name in sorted(own_attrs):
        if not _is_public_method(name):
            continue
        obj = cls.__dict__[name]

        # Unwrap classmethod / staticmethod
        actual = obj
        is_classmethod = False
        is_staticmethod = False
        if isinstance(obj, classmethod):
            actual = obj.__func__
            is_classmethod = True
        elif isinstance(obj, staticmethod):
            actual = obj.__func__
            is_staticmethod = True

        if not callable(actual) and not isinstance(actual, property):
            continue

        # Skip properties in methods list
        if isinstance(obj, property):
            continue

        method_data: dict[str, Any] = {"name": name}

        # Signature
        try:
            sig = inspect.signature(actual)
            method_data["signature"] = str(sig)
        except (ValueError, TypeError):
            method_data["signature"] = "()"

        # Docstring
        doc = inspect.getdoc(actual)
        method_data["docstring"] = doc or ""

        # Abstract
        method_data["abstract"] = getattr(actual, "__isabstractmethod__", False)
        method_data["classmethod"] = is_classmethod
        method_data["staticmethod"] = is_staticmethod

        # Parameters
        params: list[dict[str, Any]] = []
        try:
            sig = inspect.signature(actual)
            for pname, param in sig.parameters.items():
                if pname in ("self", "cls"):
                    continue
                p: dict[str, Any] = {"name": pname}
                if param.annotation is not inspect.Parameter.empty:
                    p["type"] = _format_type(param.annotation)
                if param.default is not inspect.Parameter.empty:
                    try:
                        json.dumps(param.default)
                        p["default"] = param.default
                    except (TypeError, ValueError):
                        p["default"] = repr(param.default)
                params.append(p)
        except (ValueError, TypeError):
            pass
        method_data["parameters"] = params

        # Return type
        try:
            sig = inspect.signature(actual)
            if sig.return_annotation is not inspect.Signature.empty:
                method_data["return_type"] = _format_type(sig.return_annotation)
            else:
                method_data["return_type"] = ""
        except (ValueError, TypeError):
            method_data["return_type"] = ""

        methods.append(method_data)

    return methods


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------


def _get_import_path(cls: type) -> str:
    """Build a user-facing import statement.

    Prefers the short public path from sdg_hub top-level __init__ when available.
    """
    module = cls.__module__ or ""
    name = cls.__qualname__

    # Check if it is re-exported from the blocks __init__
    try:
        import sdg_hub.core.blocks as blocks_pkg

        if hasattr(blocks_pkg, name):
            return f"from sdg_hub.core.blocks import {name}"
    except Exception:
        pass

    # Check top-level sdg_hub
    try:
        import sdg_hub

        if hasattr(sdg_hub, name):
            return f"from sdg_hub import {name}"
    except Exception:
        pass

    return f"from {module} import {name}"


def _extract_class(cls: type, source_file: str | None = None) -> dict[str, Any]:
    """Build a JSON-serializable dict describing *cls*."""
    data: dict[str, Any] = {
        "name": cls.__qualname__,
        "module": cls.__module__,
        "import_path": _get_import_path(cls),
        "docstring": inspect.getdoc(cls) or "",
        "bases": [
            b.__qualname__
            for b in cls.__mro__[1:]
            if b is not object and b.__module__ != "builtins"
        ],
    }

    # Fields
    if _is_pydantic_model(cls):
        data["fields"] = _extract_pydantic_fields(cls)
    elif _is_dataclass(cls):
        data["fields"] = _extract_dataclass_fields(cls)
    else:
        data["fields"] = []

    # Methods
    data["methods"] = _extract_methods(cls)

    # Decorator info (from AST)
    if source_file:
        dec_info = _extract_decorator_info(source_file, cls.__qualname__)
        if dec_info:
            data["decorator"] = dec_info

    return data


# ---------------------------------------------------------------------------
# Module processing
# ---------------------------------------------------------------------------


def _resolve_source_file(module: Any) -> str | None:
    """Get the source file path for a module."""
    try:
        return inspect.getfile(module)
    except (TypeError, OSError):
        return None


def _get_public_classes(module: Any, class_names: list[str] | None) -> list[type]:
    """Get classes from a module, optionally filtered by name."""
    classes: list[type] = []
    if class_names is not None:
        for name in class_names:
            cls = getattr(module, name, None)
            if cls is not None and isinstance(cls, type):
                classes.append(cls)
            elif cls is not None:
                # Might be an enum or non-type class-like object
                classes.append(cls)
            else:
                log.warning("Class %s not found in %s", name, module.__name__)
    else:
        # Auto-discover: all public classes defined in this module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name.startswith("_"):
                continue
            if obj.__module__ == module.__name__:
                classes.append(obj)
    return classes


def process_module(
    module_path: str, class_names: list[str] | None
) -> list[dict[str, Any]]:
    """Import a module and extract class information."""
    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        log.warning("Could not import %s: %s", module_path, exc)
        return []

    source_file = _resolve_source_file(mod)
    classes = _get_public_classes(mod, class_names)

    results: list[dict[str, Any]] = []
    for cls in classes:
        try:
            cls_source = None
            try:
                cls_source = inspect.getfile(cls)
            except (TypeError, OSError):
                cls_source = source_file
            data = _extract_class(cls, cls_source)
            results.append(data)
            log.info("  Extracted: %s", cls.__qualname__)
        except Exception as exc:
            log.warning(
                "  Failed to extract %s: %s", getattr(cls, "__qualname__", cls), exc
            )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for category, subcategories in MODULE_MAP.items():
        output[category] = {}
        for subcategory, modules in subcategories.items():
            log.info("Processing %s.%s ...", category, subcategory)
            entries: list[dict[str, Any]] = []
            for module_path, class_names in modules:
                entries.extend(process_module(module_path, class_names))
            output[category][subcategory] = entries

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Summary
    total_classes = sum(
        len(entries) for cat in output.values() for entries in cat.values()
    )
    total_methods = sum(
        len(e.get("methods", []))
        for cat in output.values()
        for entries in cat.values()
        for e in entries
    )
    total_fields = sum(
        len(e.get("fields", []))
        for cat in output.values()
        for entries in cat.values()
        for e in entries
    )
    log.info(
        "Done. Wrote %d classes, %d methods, %d fields to %s",
        total_classes,
        total_methods,
        total_fields,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()

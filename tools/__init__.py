"""Tool registry for VQ Agent Core.

Each tool is a plain Python function decorated with @needle.tool
(name + docstring + type annotations -> JSON schema for Needle 2).
Importing the tool modules below registers them in REGISTRY.
"""
import json

import needle

from . import web_tools, notes_tools, system_tools, decide_tools, local_tools  # noqa: F401

REGISTRY = {}


def register(fn):
    REGISTRY[fn.__name__] = fn
    return fn


# Apply registration to every @needle.tool function defined in the modules.
for _mod in (web_tools, notes_tools, system_tools, decide_tools, local_tools):
    for _name in dir(_mod):
        _fn = getattr(_mod, _name)
        if callable(_fn) and getattr(_fn, "_needle_tool", None) is not None:
            register(_fn)

del _mod, _name, _fn


def get_tool_schemas():
    """JSON schemas for Needle 2 (same list the CLI --tools flag consumes)."""
    schemas = []
    for name, fn in REGISTRY.items():
        schema = dict(getattr(fn, "_needle_tool") or needle.build_schema(fn))
        schema["name"] = name
        schemas.append(schema)
    return schemas


def dump_tools_json(path):
    """Write tools.json for `needle --tools tools.json` (Termux path)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(get_tool_schemas(), fh, indent=1)
    return path


def get(name):
    return REGISTRY.get(name)

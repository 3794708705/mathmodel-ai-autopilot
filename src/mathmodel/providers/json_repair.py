"""Shared repair helpers for OpenAI-compatible structured generation.

Every provider that asks a chat model for JSON hits the same three failure
modes: markdown fences, prose around the object, and enum values that differ
only by letter case. These helpers fix exactly those, and nothing else — a
value that is not a case-insensitive enum match still fails validation.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Type

from pydantic import BaseModel


def outermost_object(text: str) -> str:
    """Extract the outermost {...} span, dropping any prose around it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return ""
    return text[start:end + 1]


def resolve_schema(schema: dict, defs: dict) -> dict:
    """Follow a local ``$ref`` into ``$defs``."""
    if not isinstance(schema, dict):
        return {}
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return defs.get(ref[len("#/$defs/"):], {})
    return schema


def coerce_enum_casing(value: Any, schema: dict, defs: dict) -> Any:
    """Canonicalise enum strings that differ only by case.

    Models routinely answer ``"REQUIRED"`` where the schema declares
    ``"required"``. The value is already semantically correct, so map it to the
    declared member instead of failing schema validation.
    """
    if not isinstance(schema, dict):
        return value

    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return coerce_enum_casing(value, defs.get(ref[len("#/$defs/"):], {}), defs)

    for combinator in ("anyOf", "oneOf", "allOf"):
        options = schema.get(combinator)
        if isinstance(options, list):
            for option in options:
                resolved = resolve_schema(option, defs)
                if resolved.get("type") == "object" and isinstance(value, dict):
                    return coerce_enum_casing(value, option, defs)
                if resolved.get("enum") and isinstance(value, str):
                    return coerce_enum_casing(value, option, defs)

    enum = schema.get("enum")
    if isinstance(enum, list) and isinstance(value, str):
        for member in enum:
            if isinstance(member, str) and member.lower() == value.lower():
                return member
        return value

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key, item in value.items():
            if key in properties:
                value[key] = coerce_enum_casing(item, properties[key], defs)
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            for key in list(value.keys()):
                if key not in properties:
                    value[key] = coerce_enum_casing(value[key], additional, defs)
        return value

    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            return [coerce_enum_casing(item, items, defs) for item in value]
        return value

    return value


def normalise_for_schema(parsed: dict, output_schema: Type[BaseModel]) -> dict:
    """Apply safe, schema-guided normalisation to a parsed model response."""
    try:
        root = output_schema.model_json_schema()
    except Exception:
        return parsed
    defs = root.get("$defs", {}) if isinstance(root, dict) else {}
    return coerce_enum_casing(parsed, root, defs)


def parse_json_object(content: str) -> Optional[dict]:
    """Parse a JSON object from a model response, tolerating fences."""
    text = (content or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()
    for candidate in (text, outermost_object(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None

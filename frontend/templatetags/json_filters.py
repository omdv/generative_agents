"""Custom template filters for JSON handling."""

import json

from django import template
from django.utils.safestring import mark_safe

register = template.Library()


@register.filter(name="to_json")
def to_json(value):
    """Convert a Python object to a JSON string for use in templates."""
    return mark_safe(json.dumps(value))


@register.filter(name="filename")
def filename(value):
    """Convert a name to a filename-safe format (spaces to underscores)."""
    if isinstance(value, str):
        return value.replace(" ", "_")
    return value

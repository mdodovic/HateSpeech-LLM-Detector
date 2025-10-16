"""
Language configuration for selecting English or Serbian categories and prompts.
Usage:
    from language_config import get_categories_module, get_categories_prompt
"""

from typing import Literal

Language = Literal["en", "sr"]


def _import_categories(module_name: str):
    try:
        # When imported as package (examples.src)
        from . import categories_en as _en  # type: ignore
        from . import categories as _sr  # type: ignore
        return {"categories_en": _en, "categories_srb": _sr}[module_name]
    except Exception:
        # When imported as a top-level module via sys.path
        import importlib
        return importlib.import_module(module_name)


def get_categories_module(lang: Language):
    if lang == "sr":
        return _import_categories("categories_srb")
    return _import_categories("categories_en")


def get_categories_prompt(lang: Language, include_subcategories: bool = True) -> str:
    cats = get_categories_module(lang)
    return cats.get_category_prompt(include_subcategories=include_subcategories)

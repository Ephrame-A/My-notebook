"""
Chat history
============
Keeps a short rolling conversation per notebook (collection) in memory, so
follow-up questions like "what about the second one?" have context.

"""

from typing import List, Dict
from collections import defaultdict

MAX_TURNS = 6  # how many past exchanges to keep per notebook

_history: Dict[str, List[Dict[str, str]]] = defaultdict(list)


def get_history(collection_name: str) -> List[Dict[str, str]]:
    return _history[collection_name]


def add_turn(collection_name: str, role: str, content: str) -> None:
    _history[collection_name].append({"role": role, "content": content})
    _history[collection_name] = _history[collection_name][-MAX_TURNS * 2:]


def clear_history(collection_name: str) -> None:
    _history[collection_name] = []


def format_history_for_prompt(collection_name: str) -> str:
    turns = get_history(collection_name)
    if not turns:
        return ""
    lines = [f"{t['role'].capitalize()}: {t['content']}" for t in turns]
    return "Previous conversation:\n" + "\n".join(lines)

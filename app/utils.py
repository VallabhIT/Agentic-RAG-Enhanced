import json
import re
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n?(.*?)```", re.DOTALL)

_SMALLTALK_RE = re.compile(
    r"^(?:"
    r"hi|hii+|hello|hey|heya|yo|hola|namaste|howdy|sup|"
    r"good\s+(?:morning|afternoon|evening|night)|"
    r"thanks|thank\s+you|thx|ty|cheers|"
    r"ok|okay|cool|great|nice|got\s+it|"
    r"bye|goodbye|see\s+you|"
    r"how\s+are\s+you|how\s+are\s+you\s+doing|what'?s\s+up|"
    r"who\s+are\s+you|what\s+can\s+you\s+do"
    r")(?:\s+(?:there|team|bot|assistant|everyone|a\s+lot|so\s+much|very\s+much|again))*$"
)


def is_smalltalk(text: str) -> bool:
    """Cheap heuristic for obvious greetings/thanks so we can skip rewrite/router LLM calls."""
    norm = re.sub(r"[^\w\s']", " ", text.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return bool(norm) and bool(_SMALLTALK_RE.match(norm))


def strip_code_fences(text: str) -> str:
    """Return the contents of the first fenced block, or the text itself if unfenced."""
    text = text.strip()
    match = _FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


def extract_json(text: str) -> Optional[dict]:
    """Parse a JSON object from LLM output, tolerating code fences and surrounding prose."""
    text = strip_code_fences(text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def format_history(messages: list[BaseMessage], window: int) -> str:
    """Render the last `window` messages BEFORE the current user turn as plain text."""
    prior = messages[:-1][-window:] if messages else []
    lines = []
    for m in prior:
        if isinstance(m, HumanMessage):
            role = "User"
        elif isinstance(m, AIMessage):
            role = "Assistant"
        else:
            continue
        lines.append(f"{role}: {str(m.content)[:500]}")
    return "\n".join(lines) if lines else "(no prior conversation)"


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} characters]"


def to_json(obj: Any, limit: int) -> str:
    return truncate(json.dumps(obj, indent=2, default=str, ensure_ascii=False), limit)

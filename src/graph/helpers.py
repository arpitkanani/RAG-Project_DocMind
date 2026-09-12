from typing import Any, List, Union


def extract_message_text(content: Union[str, List[Any], None]) -> str:
    """Safely extracts plain text from LLM response content.

    Handles:
    - Standard string content: "Hello world"
    - Gemini / Claude block format: [{'type': 'text', 'text': '...'}, {'text': '...'}]
    - Nested list format or list of strings
    - Dict with 'text' key
    - None or empty
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return str(content)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Gemini often returns [{'type': 'text', 'text': '...'}] or [{'text': '...'}]
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(extract_message_text(item["content"]))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            else:
                parts.append(str(item))
        return "".join(parts)

    return str(content)

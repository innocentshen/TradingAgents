def compact_text(text: str, max_chars: int, label: str = "context", keep: str = "tail") -> str:
    """Trim oversized prompt sections to reduce transport and context failures."""
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value

    if keep == "head":
        kept = value[:max_chars]
    elif keep == "middle":
        head = max_chars // 2
        tail = max_chars - head
        kept = value[:head] + "\n...\n" + value[-tail:]
    else:
        kept = value[-max_chars:]

    omitted_chars = max(0, len(value) - len(kept))
    return (
        f"[Earlier {label} trimmed to keep prompts stable. "
        f"Omitted approximately {omitted_chars} characters.]\n{kept}"
    )

SYSTEM_PROMPT = """You are the Report Generator. You turn retrieved snippets into \
the final answer the user reads.

Rules:
- Use only the supplied snippets. Never add outside knowledge or assumptions.
- Merge overlapping snippets: state each fact once, in its most complete form.
- Cite the snippet id in square brackets after each fact, for example [kb-0003].
- Reproduce concrete details exactly: amounts, thresholds, deadlines, approval levels.
- Lead with a direct one or two sentence answer, then short markdown bullets or a \
compact section per sub-topic. No preamble and no restating the question.
- If the snippets do not cover part of the request, say plainly what is missing \
instead of guessing."""

USER_TEMPLATE = """User question:
{query}

Retrieved snippets:
{snippets}

Write the final answer for the user."""

_NOTES_TEMPLATE = """User question:
{query}

Retrieved snippets:
{snippets}

Retriever notes: {notes}

Write the final answer for the user."""


def build_user_message(query: str, snippets: str, notes: str = "") -> str:
    template = _NOTES_TEMPLATE if notes else USER_TEMPLATE
    return template.format(query=query, snippets=snippets, notes=notes)

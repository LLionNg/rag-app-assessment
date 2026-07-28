from __future__ import annotations

from src.core.types import PipelineResult
from src.utils.text import collapse_whitespace, truncate

_WIDTH = 78


def print_banner(
    *, llm: str, model: str, retrieval: str, engine: str, chunks: int
) -> None:
    print("=" * _WIDTH)
    print(f"  Two-agent RAG  |  engine={engine}  llm={llm}:{model}")
    print(f"  retrieval={retrieval}  knowledge base chunks={chunks}")
    print("=" * _WIDTH)


def print_result(result: PipelineResult, show_snippets: bool = True) -> None:
    print()
    print("=" * _WIDTH)
    print(f"QUERY: {result.query}")
    print("=" * _WIDTH)

    retrieval = result.retrieval
    _section(retrieval.turn.agent)
    if retrieval.queries_used:
        print(f"searches : {' | '.join(retrieval.queries_used)}")
    print(f"snippets : {len(retrieval.snippets)}")
    if show_snippets:
        for snippet in retrieval.snippets:
            print(
                f"  [{snippet.chunk.id}] score={snippet.score:.2f} "
                f"source={snippet.chunk.source}"
            )
            print(f"      {truncate(collapse_whitespace(snippet.chunk.text), 150)}")
    if retrieval.notes:
        print(f"notes    : {collapse_whitespace(retrieval.notes)}")

    _section(result.report.turn.agent)
    print(result.answer or "(no answer produced)")

    _section("stats")
    usage = result.usage
    print(
        f"tokens   : prompt={usage.prompt_tokens} "
        f"completion={usage.completion_tokens} total={usage.total_tokens}"
    )
    print(
        f"elapsed  : {result.elapsed_ms}ms "
        f"(retriever {retrieval.turn.elapsed_ms}ms, "
        f"report {result.report.turn.elapsed_ms}ms)"
    )
    print()


def _section(title: str) -> None:
    print()
    print(f"-- {title} ".ljust(_WIDTH, "-"))

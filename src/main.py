from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

from src.application import Application
from src.console import print_banner, print_result
from src.core.config import DEFAULT_CONFIG_PATH, load_settings
from src.core.exceptions import RagAppError
from src.core.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-app",
        description="Two-agent RAG: a Data Retriever feeds a Report Generator.",
    )
    parser.add_argument("query", nargs="*", help="question to answer")
    parser.add_argument(
        "-c", "--config", default=str(DEFAULT_CONFIG_PATH), help="path to config.yml"
    )
    parser.add_argument(
        "-d", "--demo", action="store_true", help="run the queries from demo.queries"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="ask questions in a loop"
    )
    parser.add_argument(
        "--no-snippets", action="store_true", help="hide the retrieved snippets"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="rebuild the embedding index instead of reusing the stored one",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="show only the report, suppressing progress logs",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="launch the web interface (the default when no query is given)",
    )
    return parser


async def run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    if args.quiet:
        settings.logging.level = "ERROR"
    configure_logging(settings.logging)
    if args.reindex:
        settings.embeddings.store.refresh = True

    async with await Application.create(settings) as app:
        print_banner(
            llm=settings.llm.provider,
            model=app.llm.model,
            retrieval=settings.retrieval.strategy,
            engine=app.orchestrator.engine,
            chunks=len(app.knowledge_base),
        )

        queries = [" ".join(args.query)] if args.query else []
        if args.demo:
            queries.extend(settings.demo.queries)

        for query in queries:
            print_result(await app.ask(query), show_snippets=not args.no_snippets)

        if args.interactive:
            await interactive_loop(app, show_snippets=not args.no_snippets)

    return 0


async def interactive_loop(app: Application, show_snippets: bool) -> None:
    print("Ask a question, or press Enter to quit.")
    while True:
        try:
            query = (await asyncio.to_thread(input, "\n> ")).strip()
        except EOFError:  # stdin closed, e.g. piped or non-interactive shell
            break
        if not query:
            break
        try:
            print_result(await app.ask(query), show_snippets=show_snippets)
        except RagAppError as exc:
            logger.error("{}", exc)


def serve_ui(args: argparse.Namespace) -> int:
    """Gradio owns its own event loop, so this path stays outside asyncio.run."""
    from src.ui.gradio_app import WebUI

    settings = load_settings(args.config)
    if args.quiet:
        settings.logging.level = "ERROR"
    configure_logging(settings.logging)
    if args.reindex:
        settings.embeddings.store.refresh = True

    WebUI(settings).launch()
    return 0


def cli() -> int:
    args = build_parser().parse_args()
    wants_cli = bool(args.query) or args.demo or args.interactive
    try:
        if args.ui or not wants_cli:
            return serve_ui(args)
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except RagAppError as exc:
        logger.error("{}", exc)
        return 1


if __name__ == "__main__":
    sys.exit(cli())

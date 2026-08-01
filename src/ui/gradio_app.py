from __future__ import annotations

import asyncio

import gradio as gr
from loguru import logger

from src.application import Application
from src.core.config import Settings
from src.core.exceptions import RagAppError
from src.core.types import PipelineResult
from src.utils.text import collapse_whitespace, truncate

_SNIPPET_PREVIEW_CHARS = 220


class WebUI:
    """Gradio front end over the same two-agent pipeline the CLI drives.

    The Application is built on the first question rather than at construction:
    it owns an httpx client and an in-process embedding model, both of which
    must belong to the event loop Gradio ends up running, not to whichever loop
    happened to be current at import time.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._app: Application | None = None
        self._lock = asyncio.Lock()

    async def _application(self) -> Application:
        async with self._lock:
            if self._app is None:
                logger.info("Building the pipeline for the first request")
                self._app = await Application.create(self.settings)
        return self._app

    async def answer(self, query: str) -> tuple[str, str, list[list[str]], str, str]:
        query = (query or "").strip()
        if not query:
            return ("_Ask something to begin._", "", [], "", "")

        try:
            application = await self._application()
            result = await application.ask(query)
        except RagAppError as exc:
            logger.error("{}", exc)
            return (f"**Error**\n\n{exc}", "", [], "", "")

        return (
            result.answer or "_The model returned nothing._",
            self._render_searches(result),
            self._render_snippets(result),
            f"**Coverage note**\n\n{result.retrieval.notes}"
            if result.retrieval.notes
            else "",
            self._render_stats(result),
        )

    @staticmethod
    def _render_searches(result: PipelineResult) -> str:
        queries = result.retrieval.queries_used
        if not queries:
            return "_No search was issued._"
        listed = "\n".join(f"{n}. `{q}`" for n, q in enumerate(queries, start=1))
        return f"**Searches issued by the Data Retriever**\n\n{listed}"

    @staticmethod
    def _render_snippets(result: PipelineResult) -> list[list[str]]:
        return [
            [
                snippet.chunk.id,
                f"{snippet.score:.2f}",
                snippet.chunk.source,
                truncate(
                    collapse_whitespace(snippet.chunk.text), _SNIPPET_PREVIEW_CHARS
                ),
            ]
            for snippet in result.retrieval.snippets
        ]

    def _render_stats(self, result: PipelineResult) -> str:
        usage = result.usage
        settings = self.settings
        return (
            f"`{settings.llm.provider}:{settings.llm.active().model}` · "
            f"retrieval `{settings.retrieval.strategy}` · "
            f"engine `{settings.orchestration.engine}` · "
            f"**{len(result.retrieval.snippets)}** snippets · "
            f"**{usage.total_tokens}** tokens "
            f"({usage.prompt_tokens} in / {usage.completion_tokens} out) · "
            f"**{result.elapsed_ms / 1000:.1f}s**"
        )

    def generate_interface(self) -> gr.Blocks:
        settings = self.settings
        with gr.Blocks(title=settings.ui.title) as interface:
            gr.Markdown(
                f"# {settings.ui.title}\n"
                "**Data Retriever** searches the knowledge base and returns raw "
                "snippets. **Report Generator** turns those snippets into a "
                "cited answer. It never answers from anything else."
            )

            with gr.Row():
                question = gr.Textbox(
                    label="Question",
                    placeholder="What is the policy on international travel?",
                    lines=2,
                    scale=5,
                )
                ask = gr.Button("Ask", variant="primary", scale=1)

            if settings.demo.queries:
                gr.Examples(
                    examples=settings.demo.queries, inputs=question, label="Try"
                )

            answer = gr.Markdown(label="Answer", value="_Ask something to begin._")

            with gr.Accordion("Retrieval trace", open=True):
                searches = gr.Markdown()
                snippets = gr.Dataframe(
                    headers=["chunk", "score", "source", "text"],
                    datatype=["str", "str", "str", "str"],
                    column_count=(4, "fixed"),
                    wrap=True,
                    label="Snippets handed to the Report Generator",
                )
                notes = gr.Markdown()

            stats = gr.Markdown()

            outputs = [answer, searches, snippets, notes, stats]
            ask.click(self.answer, inputs=question, outputs=outputs)
            question.submit(self.answer, inputs=question, outputs=outputs)

        return interface

    def launch(self) -> None:
        ui = self.settings.ui
        logger.info("Serving the UI on http://{}:{}", ui.host, ui.port)
        self.generate_interface().launch(
            server_name=ui.host,
            server_port=ui.port,
            share=ui.share,
            inbrowser=ui.open_browser,
            theme=gr.themes.Soft(),
            quiet=True,
        )

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

_THEME = gr.themes.Soft(
    font=(
        "system-ui",
        "-apple-system",
        gr.themes.Font("Segoe UI"),
        gr.themes.Font("Roboto"),
        gr.themes.Font("Helvetica Neue"),
        "sans-serif",
    ),
    font_mono=(
        "ui-monospace",
        gr.themes.Font("Cascadia Mono"),
        gr.themes.Font("SF Mono"),
        gr.themes.Font("Consolas"),
        "monospace",
    ),
)


class WebUI:
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

    async def answer(self, query: str, history: list[dict]):
        query = (query or "").strip()
        if not query:
            gr.Warning("Please enter a question.")
            yield query, history, "", [], "", ""
            return

        history = [
            *history,
            {"role": "user", "content": query},
            {"role": "assistant", "content": "Thinking..."},
        ]
        yield "", history, "", [], "", ""

        try:
            application = await self._application()
            result = await application.ask(query)
        except RagAppError as exc:
            logger.error("{}", exc)
            yield "", self._replied(history, f"**Error**\n\n{exc}"), "", [], "", ""
            return

        yield (
            "",
            self._replied(history, result.answer or "_The model returned nothing._"),
            self._render_searches(result),
            self._render_snippets(result),
            f"**Coverage note**\n\n{result.retrieval.notes}"
            if result.retrieval.notes
            else "",
            self._render_stats(result),
        )

    @staticmethod
    def _replied(history: list[dict], content: str) -> list[dict]:
        """Swap the pending assistant turn for the real one."""
        return [*history[:-1], {"role": "assistant", "content": content}]

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
            f"`{settings.llm.provider}:{settings.llm.active().model}` | "
            f"retrieval `{settings.retrieval.strategy}` | "
            f"engine `{settings.orchestration.engine}` | "
            f"**{len(result.retrieval.snippets)}** snippets | "
            f"**{usage.total_tokens}** tokens "
            f"({usage.prompt_tokens} in / {usage.completion_tokens} out) | "
            f"**{result.elapsed_ms / 1000:.1f}s**"
        )

    def generate_interface(self) -> gr.Blocks:
        settings = self.settings
        with gr.Blocks(title=settings.ui.title) as interface:
            gr.Markdown(
                f"# {settings.ui.title}\n"
                "Answers come only from the knowledge base, with citations."
            )

            chatbot = gr.Chatbot(
                label=settings.ui.title,
                height=520,
                line_breaks=True,
            )

            question = gr.Textbox(
                label="Your question",
                placeholder="Type your question here...",
                autofocus=True,
            )

            with gr.Row():
                ask = gr.Button("Ask", variant="primary", scale=2)
                reset = gr.Button("Reset conversation", variant="secondary", scale=1)

            if settings.demo.queries:
                gr.Examples(examples=settings.demo.queries, inputs=question)

            stats = gr.Markdown()

            with gr.Accordion("Retrieval trace", open=False):
                searches = gr.Markdown()
                snippets = gr.Dataframe(
                    headers=["chunk", "score", "source", "text"],
                    datatype=["str", "str", "str", "str"],
                    column_count=(4, "fixed"),
                    wrap=True,
                    label="Snippets handed to the Report Generator",
                )
                notes = gr.Markdown()

            outputs = [question, chatbot, searches, snippets, notes, stats]
            ask.click(self.answer, inputs=[question, chatbot], outputs=outputs)
            question.submit(self.answer, inputs=[question, chatbot], outputs=outputs)
            reset.click(lambda: ("", [], "", [], "", ""), outputs=outputs)

        return interface

    def launch(self) -> None:
        ui = self.settings.ui
        logger.info("Serving the UI on http://{}:{}", ui.host, ui.port)
        self.generate_interface().launch(
            server_name=ui.host,
            server_port=ui.port,
            share=ui.share,
            inbrowser=ui.open_browser,
            theme=_THEME,
            css="footer {visibility: hidden}",
            quiet=True,
        )

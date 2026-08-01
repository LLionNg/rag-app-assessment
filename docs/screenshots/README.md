# Screenshots

Capture one image per query from the web interface:

```bash
uv run rag-app
```

Ask each question under `demo.queries` in `config.yml` and screenshot the page.
Each one shows the whole pipeline in a single frame: the searches the Data
Retriever issued, the snippets it handed over with relevance scores, its
coverage note, and the Report Generator's cited answer.

Take them with a real model configured (`llm.provider: bbl_gateway`), not with
the mock placeholder - the mock's answers are stand-ins, not model output.

# Two-Agent RAG System

An agentic RAG system with two collaborating agents over a local plain-text knowledge base.

| Agent | Role | Tools | Output |
| --- | --- | --- | --- |
| **Data Retriever** | Retrieval specialist. Reformulates the request into focused searches and never answers it. | `search_knowledge_base` (custom RAG tool) | Raw, deduplicated text snippets |
| **Report Generator** | Writer and synthesiser. Turns the snippets into the final answer. | none | Cited, non-redundant markdown answer |

Orchestration is a sequential [LangGraph](https://langchain-ai.github.io/langgraph/) workflow: the Data Retriever's output is the only input the Report Generator receives.

```
                ┌──────────────────┐   snippets    ┌───────────────────┐
 query ────────►│  Data Retriever  │──────────────►│  Report Generator │────► answer
                └────────┬─────────┘               └───────────────────┘
                         │ tool call
                         ▼
              search_knowledge_base
                         │
                  ┌──────┴───────┐
                  │  Retriever   │  BM25 / embeddings / hybrid (RRF)
                  └──────┬───────┘
                         │
                 knowledge_base.txt ──► chunked at load time
```

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
```

```bash
python -m src.main "What is the policy on international travel?"
```

The default configuration uses the **mock LLM provider**, so this runs with no API key and no network access. It exercises the real agent loop, the real tool call, and the real retriever — only the model's wording is a placeholder.

Other entry points:

```bash
python -m src.main --demo
```

```bash
python -m src.main --interactive
```

```bash
pytest
```

## Connecting a real model

1. Put the key in `.env` (see `.env.example`) — never in `config.yml`:
   ```
   AZURE_OPENAI_API_KEY=...
   ```
2. Switch the provider in `config.yml`:
   ```yaml
   llm:
     provider: azure_openai
   ```

The `azure_openai` block is pre-filled for the `gpt-5-mini` deployment named in the test brief. Two quirks of that model family are configuration flags rather than code branches: `token_param: max_completion_tokens` and `supports_temperature: false`.

To use any OpenAI-compatible endpoint instead, set `llm.provider: openai` and point `base_url` at it.

## Configuration

Everything is driven by [`config.yml`](config.yml); secrets are read from the environment via the `api_key_env` indirection. Values support `${VAR}` and `${VAR:-default}` expansion. Point at a different file with `--config path/to/config.yml`.

| Section | What it controls |
| --- | --- |
| `logging` | Level, rotating file sink, JSON serialisation |
| `llm` | Active provider, temperature, token cap, timeout, retries, per-provider connection settings |
| `embeddings` | Optional embedding provider (`null` disables it) |
| `knowledge_base` | Source file and chunking strategy (`paragraph` \| `fixed`), sizes, overlap |
| `retrieval` | Strategy (`keyword` \| `semantic` \| `hybrid`), `top_k`, score floor, BM25 `k1`/`b`, stopwords, RRF settings |
| `agents` | Agent display names, the Data Retriever's tool-call budget, snippet caps |
| `orchestration` | Workflow engine |
| `demo` | Queries used by `--demo` |

### Retrieval strategies

- **`keyword`** (default) — in-memory BM25. No external service, works offline.
- **`semantic`** — cosine similarity over embedded chunks. Requires `embeddings.provider`.
- **`hybrid`** — weighted reciprocal rank fusion of the two.

Scores are normalised to 0–1 so `retrieval.min_score` means the same thing whichever strategy is active. Chunks with no term overlap at all are dropped before normalisation, so an off-topic question correctly returns nothing.

## Layout

```
src/
├── main.py              CLI entry point
├── application.py       composition root: builds and wires every component
├── console.py           terminal rendering
├── core/                config, logging, shared types, exceptions
├── llm/                 LLMProvider base + azure_openai, openai, mock
├── embeddings/          EmbeddingProvider base + openai/azure implementations
├── retrieval/           Retriever base + keyword, semantic, hybrid; chunking; knowledge base
├── tools/               Tool base + search_knowledge_base
├── prompts/             one module per agent
├── agents/              BaseAgent (tool-calling loop) + the two agents
└── orchestration/       Orchestrator base + LangGraph workflow
data/knowledge_base.txt  sample corpus
tests/                   chunking, retrieval, agents, end-to-end
```

Each pluggable concern is an abstract base class with a factory: `LLMProvider`, `EmbeddingProvider`, `Retriever`, `Chunker`, `Tool`, `BaseAgent`, `Orchestrator`. Adding a provider, a retrieval strategy, or a tool means adding one subclass and one registry entry — no changes to the agents or the workflow.

Shared behaviour lives in the bases: retries with exponential backoff and latency logging in `LLMProvider`, batching and concurrency in `EmbeddingProvider`, ranking and score normalisation in `Retriever`, the tool-calling loop with concurrent tool execution in `BaseAgent`.

## How the agent loop works

`BaseAgent._converse` runs the model, executes any requested tool calls concurrently with `asyncio.gather`, feeds the results back, and repeats. Once the configured tool budget is spent the tools are withdrawn from the request, which forces the model to produce a final answer instead of looping — the loop cannot run away.

The Data Retriever forwards the tool's **structured** output rather than the model's retelling of it, so snippets reach the Report Generator byte-for-byte as they appear in the knowledge base. The model's job is query formulation and relevance judgement, not transcription.

## Branches

| Branch | Orchestration |
| --- | --- |
| `main` | LangGraph `StateGraph` |
| `framework-free` | Plain `asyncio`, no LangChain/LangGraph dependency |

Only the orchestration layer differs. Everything else — providers, retrieval, tools, agents, prompts, config — is identical, which is the point of keeping the framework at the edge of the design.

## Sample output

Screenshots for several queries are in [`docs/screenshots/`](docs/screenshots/).

## Notes and scope

- Streaming responses, a conversation store, and an HTTP API were left out deliberately: the brief is a single-shot two-agent workflow, and none of them would show more about retrieval or orchestration.
- The knowledge base is fictional sample data written for this exercise.

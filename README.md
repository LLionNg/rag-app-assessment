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

The project is uv-managed. `uv sync` provisions the interpreter pinned in `.python-version`, creates `.venv`, and installs the exact versions in `uv.lock` plus the project itself:

```bash
uv sync --extra bge-flag
```

```bash
uv run rag-app "What is the policy on international travel?"
```

The default configuration is fully local: **BGE-M3 embeddings** running in-process and the **mock LLM provider**. No API key, and no network access after the first run. It exercises the real agent loop, the real tool call and the real vector retrieval — only the model's wording is a placeholder.

That extra pulls torch, and the first run downloads the BGE-M3 weights into the HuggingFace cache (~4 GB on disk). To skip both, switch to keyword retrieval — plain `uv sync` is then enough:

```yaml
embeddings: { provider: null }
retrieval:  { strategy: keyword }
```

| Command | What it does |
| --- | --- |
| `uv run rag-app "<question>"` | answer one question |
| `uv run rag-app --demo` | run every query under `demo.queries` |
| `uv run rag-app --interactive` | ask questions in a loop |
| `uv run rag-app --no-snippets` | hide the retrieval trace |
| `uv run rag-app --reindex` | rebuild the embedding index instead of reusing it |
| `uv run rag-app -c other.yml ...` | use a different config file |
| `uv run pytest` | run the tests |
| `uv run ruff check src tests` | lint |
| `uv run ruff format src tests` | format |

Every command runs inside the locked environment, so there is nothing to activate. `uv run python -m src.main ...` is equivalent to `uv run rag-app ...`.

## The assessment gateway (gpt-5-mini)

The default `llm.provider` is `bbl_gateway`. Put the key in `.env` and it runs:

```
BBL_LLM_API_KEY=...
```

That endpoint is **not** Azure OpenAI chat-completions. It is the **Responses API** behind an Azure API Management gateway, which differs in four ways that the provider has to handle:

| | Chat Completions | Responses API |
| --- | --- | --- |
| System prompt | a `system` message | top-level `instructions` |
| History | `messages` | flat `input` list of typed items |
| Tool schema | nested `{"function": {...}}` | flat `{"type","name","parameters"}` |
| Tool result | `{"role":"tool","tool_call_id"}` | `{"type":"function_call_output","call_id"}` |

The subtle one: gpt-5-mini is a reasoning model, and a `function_call` replayed **without the `reasoning` item it was emitted with** is rejected outright. `Message.raw_items` carries the provider's own output items back verbatim so the round trip survives, which is why the abstraction has an escape hatch for provider-native payloads.

### Living inside 1000 tokens/minute

The grant is metered, and gpt-5-mini bills hidden reasoning tokens — a three-word answer cost 192 of them at default effort. Four settings keep a full query inside roughly 3,600 tokens:

- `reasoning_effort: low` — reasoning tokens per call drop about 3×
- `parallel_tool_calls: false` and `max_tool_calls: 1` — every parallel search replays its **entire** result into the next request; five searches at once cost 4,302 tokens on a single call
- `top_k` is not exposed in the tool schema, so the model cannot raise the snippet count past the configured budget
- `max_snippet_chars: 700` — one full chunk, without the metered tail

On a 429 the provider honours the gateway's `Retry-After` (60s) rather than an exponential backoff that starts in milliseconds and could never clear a per-minute window.

Note the gateway's `x-ratelimit-limit-tokens` header advertises 250,000/min — that reflects the upstream Azure deployment, not the per-key policy. The 1,000/min figure in the grant email is the one that actually bites.

## Connecting a different model

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
| `embeddings` | Provider (`null` disables it), batch size, expected `dimensions`, local-model `options` |
| `embeddings.store` | Where the `.npz` index lives, and whether to reuse or rebuild it |
| `knowledge_base` | Source file and chunking strategy (`paragraph` \| `fixed`), sizes, overlap |
| `retrieval` | Strategy (`keyword` \| `semantic` \| `hybrid`), `top_k`, score floor, BM25 `k1`/`b`, stopwords, RRF settings |
| `agents` | Agent display names, the Data Retriever's tool-call budget, snippet caps |
| `orchestration` | Workflow engine |
| `demo` | Queries used by `--demo` |

### Retrieval strategies

- **`semantic`** (default) — cosine similarity over BGE-M3 embeddings. Handles paraphrase.
- **`keyword`** — in-memory BM25. No model, no download, exact-term precision.
- **`hybrid`** — weighted reciprocal rank fusion of the two.

BM25 scores are normalised to 0–1, so `retrieval.min_score` acts as a relative floor there. Cosine values are kept raw and gated by `retrieval.semantic.min_similarity` instead. Chunks with no signal at all are dropped before ranking, so an off-topic question correctly returns nothing.

**`min_similarity` is model-specific and must be re-measured if the embedding model changes.** An absolute cosine floor does not transfer between models. On this corpus BGE-M3 puts on-topic hits at 0.46–0.73 and off-topic queries at or below 0.33, so 0.40 sits in the middle of a clean gap: every relevant chunk survives and nothing irrelevant leaks. A floor of 0.60 — plausible-looking, and correct for some other models — would admit only 1 of 16 chunks and silently starve the Report Generator no matter what `top_k` said.

Why `hybrid` is worth considering: BM25 alone cannot bridge vocabulary. Ask *"How do I get money back after a trip overseas?"* and it misses the reimbursement section entirely — `money`, `overseas` and `abroad` appear nowhere in the corpus, while `trip` appears inside the corporate-card section, which then ranks first. Embeddings fix that. Conversely BM25 is sharper on the exact tokens this corpus is full of — `USD 220`, `Tier 1`, `grade 5`, named portals — which embeddings blur.

## Embedding pipeline

Everything runs locally through [`BGEM3EmbeddingProvider`](src/embeddings/bge_m3.py); there is no vector database.

1. `knowledge_base.txt` is chunked (16 paragraph chunks by default)
2. Chunks are embedded in batches to **1024-dim** dense vectors. Encoding is blocking work, so it runs in a worker thread behind a lock — one in-process model must not be entered concurrently
3. Vectors are written verbatim to `data/index/knowledge_base.npz`, beside a Pydantic-validated `knowledge_base.json` manifest
4. Later runs load the `.npz` and skip embedding entirely

The manifest carries an `IndexFingerprint` — model name, dimensions, chunk count, and a SHA-256 digest of the exact chunk text. It is compared by equality, so editing the knowledge base, changing the chunking strategy, or switching embedding model all invalidate the index automatically. Force a rebuild with `--reindex`.

Two Pydantic guards make the 1024-dim contract explicit rather than assumed:

- `embeddings.providers.bge_m3.dimensions: 1024` is checked against **every** batch in `EmbeddingProvider._assert_dimensions`, so a misconfigured model raises instead of silently writing vectors of the wrong width
- `EmbeddingManifest` has a `model_validator` rejecting a manifest whose chunk-id list disagrees with its fingerprint

Cosine similarity is our own function in [`similarity.py`](src/retrieval/similarity.py) — `dot(a, b) / (‖a‖·‖b‖)`, the same formula the reference service uses, evaluated for the whole corpus in one matrix product and unit-tested against a row-by-row implementation.

### Swapping the backend

`options.backend` selects how BGE-M3 is loaded, and both produce the same 1024-dim dense vectors:

| Backend | Install | Notes |
| --- | --- | --- |
| `flag_embedding` (default) | `uv sync --extra bge-flag` | `BGEM3FlagModel`, the loader the reference service uses |
| `sentence_transformers` | `uv sync --extra bge` | Lighter dependency tree |

Hosted embeddings remain available by pointing `embeddings.provider` at `azure_openai` or `openai` — the retriever, the store and the similarity function are unchanged, only the provider differs.

## Branches

| Branch | Orchestration |
| --- | --- |
| `main` | LangGraph `StateGraph` |
| `framework-free` | Plain `asyncio`, no LangChain/LangGraph dependency |

Only the orchestration layer differs. Everything else — providers, retrieval, tools, agents, prompts, config — is identical, which is the point of keeping the framework at the edge of the design.

## Sample output

`docs/screenshots/` is where the per-query screenshots go. They should be captured with a real model configured — the mock provider's answers are placeholders and would not show the synthesis quality the Report Generator is judged on.

## Notes and scope

- Streaming responses, a conversation store, and an HTTP API were left out deliberately: the brief is a single-shot two-agent workflow, and none of them would show more about retrieval or orchestration.
- The knowledge base is fictional sample data written for this exercise.

# State Tax Form Assistant Agent

An AI assistant that answers questions about US state tax forms using
Retrieval Augmented Generation (RAG) over official, public form instructions.

A LangGraph agent with hybrid retrieval, 4 tools, conversation memory, an
automated evaluation suite, a REST API and Docker packaging.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your API key
```

## Add forms

Download instruction PDFs from official state revenue department websites
(public documents only) into `data/pdfs/`, named `STATE_FORM.pdf`:

```
data/pdfs/NE_1040N-instructions.pdf
data/pdfs/NC_D-400-instructions.pdf
data/pdfs/WI_Form1-instructions.pdf
```

## Run

```bash
python -m src.ingest              # build the vector store
python -m src.ask                 # ask questions
python -m src.ask --state NC      # search one state only
python -m src.chat                # chat with the agent (tools + memory)
```

## How it works

1. **Ingest:** PDFs are split into ~800-character chunks, tagged with state,
   form and page, embedded locally with `all-MiniLM-L6-v2`, and stored in ChromaDB.
2. **Retrieve:** the top 5 most similar chunks are found (optionally filtered by state).
3. **Answer:** the LLM answers only from those chunks and cites `[STATE form, p.N]`,
   or says it couldn't find the answer instead of guessing.

## Agent (Day 2)

```
START -> agent --(tool calls?)--> tools -> agent -> ... -> END
```

Built as a LangGraph `StateGraph`. The LLM decides which tools to call; the
graph runs them and loops back until it has a final answer. A `MemorySaver`
checkpointer keeps each conversation thread, so follow-up questions work.

| Tool | Purpose |
|---|---|
| `search_tax_instructions` | Hybrid (BM25 + vector) search over the loaded instructions |
| `list_loaded_forms` | Lists which states and forms are available |
| `calculate` | Safe arithmetic (parses the expression; never uses `eval`) |
| `validate_check_digit` | Modulus 11 check-digit validation for scanline numbers |

## Evaluation (Day 3)

`eval/questions.json` holds 30 test cases with known answers from the
instructions: 21 retrieval questions, calculator, combined search + calculation,
check-digit, form listing, questions the agent must refuse, and a two-turn
memory test.

```bash
python -m src.evaluate              # run everything
python -m src.evaluate --delay 4    # slow down for free-tier rate limits
```

Each answer is scored automatically on:

- **keywords**: contains the expected facts (whole-word matching) and none of the forbidden ones
- **citation**: cites at least one correct page
- **tools**: the agent called the expected tools

Results are saved to `eval/results/` after every question, so a run that hits
a rate limit can continue with `--resume`.

### Results

All runs scored with the same grader, using `openai/gpt-oss-120b` on Groq.

| Run | Change | Passed | Keywords | Citations | Tools |
|---|---|---|---|---|---|
| 1 | Baseline | 25/30 (83%) | 90% | 88% | 96% |
| 2 | Stricter prompt rules, page-range chunk labels | 27/30 (90%) | 97% | 92% | 100% |

Fixes along the way: PDF ligature cleanup ("ﬁ le" → "file"), chunking whole
documents so lists crossing page breaks stay together, hybrid BM25 + vector
search, and grader fixes for Unicode hyphens and citation formats.

**Known limitations:** one question names "the payment coupon" instead of
form TC-547, and answers vary slightly between runs, so scores can move by
one question (about 3 points).

## Web app

```bash
uvicorn src.api:app --reload
```

Open http://localhost:8000 for a chat interface:

- Ask questions in plain words; each answer shows which steps the agent took
  and clickable citation chips that open the PDF at the cited page.
- Stop a question that's taking too long, or start a new chat.
- Add a new PDF (state + form name) from the sidebar. It is read, split and
  indexed immediately, then the agent can answer from it. Re-uploading the
  same state and form replaces it; documents can also be removed.
- Limit answers to one document with the "Answer from" menu.

Scanned PDFs with no selectable text are rejected with a clear message
(OCR is not supported yet).

## REST API (Day 4)

```bash
uvicorn src.api:app --reload
```

Open http://localhost:8000/docs for interactive documentation.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Status, chunk count, active model |
| GET | `/forms` | Loaded states and forms |
| POST | `/search` | Retrieval only, no LLM call: `{"query": "...", "state": "UT"}` |
| POST | `/ask` | Ask the agent: `{"question": "...", "thread_id": "optional", "document": "optional"}` |
| GET | `/documents` | Indexed documents with page and chunk counts |
| POST | `/documents` | Upload a PDF (multipart: `file`, `state`, `form`) |
| DELETE | `/documents/{source_file}` | Remove a document |

Pass the `thread_id` from one `/ask` response into the next to continue the
same conversation.

## Docker

```bash
docker compose up --build
```

The image uses CPU-only PyTorch and downloads the embedding model at build
time. On first start the container builds its vector store from
`data/pdfs/`; a named volume keeps it between restarts. API keys come from
`.env` at runtime and are never copied into the image.

## Tests

```bash
python -m pytest
```

Unit tests cover the calculator (including blocking code injection), the
check-digit validator, and the API (validation, conversation threads, rate-limit
handling) with the agent replaced by a fake, so no API key is needed.
# TokenLimitTask — Conversation Shrinking Pipeline (Python 3.12.0)

This project implements an automated conversation-shrinking pipeline that trims oversized, role-based chat histories to fit within a model's context window while preserving essential intent and technical fidelity. It also provides tooling to test the shrunk chat directly with the Gemini API and persist the continued conversation.

Key files:

-   `chat_shrink.py` — Core pipeline with optimized chunking and summarization
-   `example_usage.py` — Runs the shrinker on `largeInput.json` and saves results
-   `test_shrunk_chat.py` — Sends shrunk chats directly to Gemini and saves continuations
-   `gemini_api.py` — Simple adapter for calling Gemini with a role-based message list

## Features

-   Preserves conversation flow and critical context:
    -   Keeps the first `system` message intact
    -   Keeps the latest `user` message
    -   Includes prior `assistant` messages (summarized when necessary)
-   Token-aware pipeline with two limits per model:
    -   `max_tokens` (context window size)
    -   `output_tokens` (maximum generation window)
-   Optimized chunking and batching to reduce API calls
-   Protection and restoration of structured content (code blocks, JSON, tables)
-   Model-aware token counting using `tiktoken` (fallback to `cl100k_base`)
-   Organized outputs:
    -   `shrunk_chats/` contains shrunk chat JSON files
    -   `continued_chats/` contains continued conversations after testing with Gemini
-   Rich metadata written to each output file (see details below)

## Requirements

-   Python 3.12.0

-   Dependencies (installed via `requirements.txt`):
    -   `tiktoken`
    -   `requests`
    -   `python-dotenv`

Gemini API key provided via environment variable:

-   `.env` with `LLM_API_KEY=your_gemini_api_key`

## Install & Setup (Windows PowerShell)

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
Copy-Item .env.example .env -ErrorAction SilentlyContinue  # optional if example exists
# Then edit .env and set LLM_API_KEY
```

## Input format

`largeInput.json` should be a list of messages or an object with `data` key:

```json
[
	{ "role": "system", "content": "..." },
	{ "role": "user", "content": "..." },
	{ "role": "assistant", "content": "..." }
]
```

or

```json
{ "data": [ { "role": "system", "content": "..." }, ... ] }
```

## How it works (high level)

The primary function is:

```python
from chat_shrink import shrink_chat

shrunk = shrink_chat(
        chat_history=messages,
        max_tokens=128000,        # model context window
        output_tokens=4000,       # model max generation window
        model_name="gpt-4o"      # used for token counting & summarization
)
```

Pipeline steps:

1. Validate and extract the first system message.
2. Keep the most recent user message.
3. Include assistant messages (summarize them first if over budget).
4. If still over budget, summarize the last user message.
5. Chunking uses `max_tokens` and `output_tokens` to size batches efficiently.
6. Protected sections (code blocks, JSON, tables) are extracted and restored to preserve fidelity.

## Run the shrinker

`example_usage.py` loads `largeInput.json`, runs the pipeline for multiple models, and saves outputs to `shrunk_chats/`.

```powershell
python .\example_usage.py
```

It uses per-model settings like:

-   Gemini 2.0 Flash: `context_window=1_000_000`, `output_window=8000`
-   GPT-4o: `context_window=128_000`, `output_window=4000`
-   GPT-4: `context_window=32_000`, `output_window=4000`

Each run writes a JSON file to `shrunk_chats/` with metadata such as:

-   `original_messages`, `shrunk_messages`
-   `model_name`, `model_context_window`, `model_output_window`
-   `processing_time_seconds`
-   `estimated_original_tokens`, `estimated_shrunk_tokens`
-   `token_reduction_percentage`
-   `timestamp`

## Test the shrunk chats with Gemini

`test_shrunk_chat.py` loads shrunk chats from `shrunk_chats/`, sends the chat as-is to Gemini (no added test prompts), prints the response, and saves the continued chat to `continued_chats/`.

```powershell
python .\test_shrunk_chat.py
```

The saved continuation includes metadata:

-   `source_file`, `test_number`, `total_messages`
-   `original_messages`, `new_messages`
-   `original_metadata` (all metadata from the shrunk source file)
-   `model_name`, `model_context_window`, `model_output_window`
-   `timestamp`, `source_file_created`

## Outputs & directories

-   `shrunk_chats/shrunk_chat_<model>_<context>.json`
-   `continued_chats/<source>_continued_test_<n>.json`

This keeps the initial shrinking artifacts separate from follow-up conversations.

## Notes & tips

-   The `output_tokens` parameter is key to faster summarization: it bounds generation and informs chunk sizing to minimize calls.
-   Token counts use `tiktoken`. For unknown models we default to `cl100k_base` as a reasonable approximation.
-   Large inputs may still take time to summarize; consider tuning model configs or simplifying instructions if performance is critical.

## Troubleshooting

-   Ensure `.env` has a valid `LLM_API_KEY` for Gemini before running tests.
-   If no files are found during testing, run `example_usage.py` first; outputs are written to `shrunk_chats/`.
-   On Windows, run scripts from an activated virtual environment to ensure dependencies are available.

## License

This repository is provided as-is for demonstration and integration purposes.

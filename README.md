# Miru Manhwa Comment Agent

Standalone tooling for a Miru manhwa comment simulation workflow.

It includes:
- `generate_manhwa_ai_comments.py` to generate reader-style comment ideas from chapter images with Qwen Vision
- `simulate_manhwa_comment_agents.py` to create a fresh Miru account every cycle, generate a comment, and post it to the next chapter thread
- `setup_manhwa_agent.sh` to create a virtualenv, install dependencies, and bootstrap `.env`

## Requirements

- Python 3.10+
- A running Miru backend with manga routes enabled
- `QWEN_API_KEY` for Qwen Vision image analysis

## Setup

```bash
./setup_manhwa_agent.sh
```

That creates `.venv`, installs `requirements.txt`, and copies `.env.example` to `.env` if needed.

The simulator uses only your backend API for registration, login, and comment posting. It does not call Supabase directly.
The vision request path is OpenAI-compatible and defaults to Qwen Vision.

## Generate Comments Only

```bash
./.venv/bin/python generate_manhwa_ai_comments.py \
  --series-slug YOUR_MANHWA_SLUG \
  --language en \
  --sample-pages 1
```

## Run Sequential Chapter Simulation

This starts at the first available chapter and moves forward one chapter at a time. Progress is stored in `.manhwa-agent-state.json`.

```bash
./.venv/bin/python simulate_manhwa_comment_agents.py \
  --series-slug YOUR_MANHWA_SLUG \
  --language en \
  --interval-minutes 10
```

To keep posting with different users on the same chapter before advancing:

```bash
./.venv/bin/python simulate_manhwa_comment_agents.py \
  --series-slug YOUR_MANHWA_SLUG \
  --language en \
  --interval-minutes 10 \
  --comments-per-chapter 3
```

To restart from chapter 1, delete `.manhwa-agent-state.json`.

## Tests

```bash
./.venv/bin/python -m pytest -q
```

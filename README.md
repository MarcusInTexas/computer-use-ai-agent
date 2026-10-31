# Computer-use AI agent

A Python computer-use research system built around Playwright. At startup the coordinator accepts a topic from the user and delegates it to `SearchAgent`, which searches the public web through a browser, observes the rendered results, and reports a configurable number of related URLs.

## Setup

```bash
cd ~/Desktop/safc-agent/computer-use-ai-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Run

Run the coordinator, which is the application entry point:

```bash
python3 coordinator.py
```

Headless:

```bash
python3 coordinator.py --headless "computer-use AI agents"
```

SearchAgent finds 10 URLs by default. Change the count with `--count`:

```bash
python3 searchAgent.py --headless --count 20 "computer-use AI agents"
```

When successful, it prints:

```text
SearchAgent has found:
<URL 1>
<URL 2>
...
```

The coordinator currently registers SearchAgent as its first research agent. Additional specialized agents can be added through the `ResearchAgent` protocol.

## Architecture

- `coordinator.py` — application entry point and agent coordinator
- `searchAgent.py` — Playwright web-search agent
- `rankingAgent.py` — Playwright page reader and LLM-based relevance/trust evaluator
- `agent.py` — original SAFC browser proof of concept

SearchAgent uses a browser-based Bing search, rather than calling a search API. It keeps the result count configurable through `SearchAgent(result_count=...)` or the `--count` command-line option. Before returning results, it sends only the Bing result names—not the referenced page contents—to an OpenAI-compatible LLM endpoint for relevance ordering. The coordinator passes the selected URLs to RankingAgent, which separately visits each page and evaluates relevance and trustworthiness. Set `OPENAI_API_KEY` before running the coordinator; optionally set `OPENAI_MODEL`, `OPENAI_BASE_URL`, and `LLM_MAX_RETRIES` (default: 3).

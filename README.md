# SAFC computer-use agent

A small Python computer-use agent built around Playwright. It opens the San Antonio FC website, observes the rendered page, reasons about what to do, acts through the browser, and extracts the next scheduled match.

## Setup

```bash
cd ~/Desktop/safc-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Run

Show the browser:

```bash
python agent.py
```

Headless:

```bash
python agent.py --headless
```

The current reasoner is deterministic and intentionally transparent. To add more complicated capabilities, implement the `Reasoner` protocol and return actions from the small `Action` vocabulary; keep the `Agent.run()` observe → reason → act loop unchanged.

## Notes from episode 371 applied here

- Playwright is the computer-use layer: the agent uses the same rendered interface a person sees.
- `Observation` is the context boundary between browser and reasoner.
- `Reasoner` is separated from `BrowserComputer`, making an LLM reasoner replaceable later.
- The agent has a step limit and a deliberately small action vocabulary, useful safeguards as autonomy grows.
- The site schedule is the source of truth; no private sports API is used.

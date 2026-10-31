"""A small, extensible computer-use agent with a user-provided task.

The important part is the observe -> reason -> act loop. The initial reasoner
is deliberately deterministic: it uses rendered page text, not a private API.
A future LLM reasoner can implement the same Reasoner protocol and handle
arbitrary tasks supplied by the user.
"""
from __future__ import annotations

import argparse
import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Protocol

from playwright.async_api import Browser, Page, async_playwright

SCHEDULE_URL = "https://www.sanantoniofc.com/2026-schedule/"
TEAM_NAME = "San Antonio FC"


@dataclass
class Observation:
    """Everything the reasoner is allowed to see at one loop iteration."""
    url: str
    title: str
    text: str
    links: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class Action:
    """A small, safe action vocabulary; extend this as capabilities grow."""
    kind: str
    target: Optional[str] = None
    value: Optional[str] = None


class Reasoner(Protocol):
    def decide(self, observation: Observation) -> Action: ...


class ScheduleReasoner:
    """Handle the initial SAFC task from visible schedule text."""
    def decide(self, observation: Observation) -> Action:
        if "2026 Schedule" not in observation.text:
            return Action("navigate", SCHEDULE_URL)
        return Action("finish")


class TaskReasoner:
    """Task-aware reasoner placeholder for the next LLM-backed iteration.

    The task is captured now so the public agent interface is no longer tied
    to a hard-coded target. Until a general reasoner is added, the known SAFC
    task uses the deterministic implementation and other tasks fail clearly.
    """
    def __init__(self, task: str):
        self.task = task

    def decide(self, observation: Observation) -> Action:
        if self.task.lower() in {
            "find san antonio fc's next game",
            "find san antonio fc’s next game",
            "find the next san antonio fc game",
        }:
            return ScheduleReasoner().decide(observation)
        raise NotImplementedError(
            "General text tasks require an LLM reasoner; the initial supported "
            "task is finding San Antonio FC's next game."
        )


class BrowserComputer:
    """Computer-use surface exposed to the agent."""
    def __init__(self, page: Page):
        self.page = page

    async def observe(self) -> Observation:
        return Observation(
            url=self.page.url,
            title=await self.page.title(),
            text=await self.page.locator("body").inner_text(),
            links=await self.page.locator("a").evaluate_all(
                "els => els.map(a => ({text: (a.innerText || '').trim(), href: a.href}))"
            ),
        )

    async def act(self, action: Action) -> None:
        if action.kind == "navigate":
            await self.page.goto(action.target or SCHEDULE_URL, wait_until="domcontentloaded")
            await self.page.wait_for_timeout(1500)
        elif action.kind == "finish":
            return
        else:
            raise ValueError(f"Unsupported action: {action.kind}")


class Agent:
    def __init__(self, computer: BrowserComputer, reasoner: Reasoner, max_steps: int = 5):
        self.computer, self.reasoner, self.max_steps = computer, reasoner, max_steps

    async def run(self) -> Observation:
        """Run the reusable observe -> reason -> act loop."""
        for step in range(1, self.max_steps + 1):
            observation = await self.computer.observe()
            action = self.reasoner.decide(observation)
            print(f"[step {step}] observe={len(observation.text)} chars; act={action.kind}")
            if action.kind == "finish":
                return observation
            await self.computer.act(action)
        raise RuntimeError("Agent reached max_steps without completing its task")


def _clean_lines(text: str) -> List[str]:
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]


def find_next_match(text: str, today: Optional[date] = None) -> Dict[str, str]:
    """Extract the first not-yet-played schedule card from rendered text.

    The site's cards currently render as HOME/AWAY, weekday, month/day, time,
    opponent, and venue.  This parser tolerates extra navigation text and
    keeps the extraction independent of fragile CSS class names.
    """
    today = today or date.today()
    lines = _clean_lines(text)
    months = {m: i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1
    )}
    for i, line in enumerate(lines):
        match = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})", line)
        if match and match.group(1) in months:
            month, day = months[match.group(1)], int(match.group(2))
            date_index = i
        else:
            match = re.fullmatch(r"(\d{1,2})", line)
            if not match or i == 0 or lines[i - 1] not in months:
                continue
            month, day = months[lines[i - 1]], int(match.group(1))
            date_index = i
        year = today.year
        try:
            event_date = date(year, month, day)
        except ValueError:
            continue
        if event_date < today:
            continue
        time = next((x for x in lines[date_index + 1:date_index + 5] if re.fullmatch(r"\d{1,2}:\d{2} [AP]M", x)), "")
        time_index = date_index + 1 + next((j for j, x in enumerate(lines[date_index + 1:date_index + 5]) if re.fullmatch(r"\d{1,2}:\d{2} [AP]M", x)), 0)
        opponent = next((x for x in lines[time_index + 1:time_index + 5] if x not in {"HOME", "AWAY", "TICKETS"} and not x.startswith("at ")), "")
        venue = next((x for x in lines[time_index + 1:time_index + 7] if x.startswith("at ") or x.startswith("vs. ")), "")
        month_name = next(name for name, number in months.items() if number == month)
        weekday = lines[date_index - 1] if date_index > 0 and lines[date_index - 1] in {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"} else ""
        display_date = f"{weekday} {month_name} {day}, {year}".strip()
        return {"date": event_date.isoformat(), "display_date": display_date, "time": time, "opponent": opponent, "venue": venue}
    raise LookupError("No upcoming match was found in the rendered schedule")


async def main(task: str, headless: bool = False) -> Optional[Dict[str, str]]:
    """Run the browser agent for a task entered by the user."""
    async with async_playwright() as playwright:
        browser: Browser = await playwright.chromium.launch(headless=headless)
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})
        await page.goto("https://www.sanantoniofc.com", wait_until="domcontentloaded")
        computer = BrowserComputer(page)
        observation = await Agent(computer, TaskReasoner(task)).run()
        result = None
        if task.lower() in {
            "find san antonio fc's next game",
            "find san antonio fc’s next game",
            "find the next san antonio fc game",
        }:
            result = find_next_match(observation.text)
            print(f"\nNext {TEAM_NAME} game: {result['display_date']} at {result['time']}")
            print(f"Opponent: {result['opponent']}")
            if result["venue"]:
                print(f"Venue: {result['venue']}")
        await browser.close()
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computer-use AI agent")
    parser.add_argument("--headless", action="store_true", help="run without showing the browser")
    parser.add_argument("task", nargs="?", help="task to perform; otherwise you will be prompted")
    args = parser.parse_args()
    task = args.task or input("What topic would you like to research? ").strip()
    if not task:
        parser.error("a non-empty task is required")
    print(f"Task: {task}")
    asyncio.run(main(task, headless=args.headless))

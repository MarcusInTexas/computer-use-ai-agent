"""Application entry point and coordinator for multiple research agents.

The coordinator owns user interaction and delegation. Individual agents should
focus on one research job and return a result; adding another agent later
should not require changing the command-line interface.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from rankingAgent import RankingAgent
from searchAgent import SearchAgent


@dataclass
class AgentResult:
    """A normalized result returned by a coordinated agent."""
    agent_name: str
    task: str
    result: Any = None
    error: Optional[str] = None


class ResearchAgent(Protocol):
    name: str

    async def research(self, topic: str, headless: bool = False) -> AgentResult: ...


class RankingResearchAgent:
    """Coordinator adapter for the LLM-backed RankingAgent."""

    name = RankingAgent.name

    def __init__(self, urls: List[str]):
        self.urls = urls

    async def research(self, topic: str, headless: bool = False) -> AgentResult:
        try:
            result = await RankingAgent().research(topic, self.urls, headless=headless)
            return AgentResult(self.name, topic, result=result)
        except Exception as exc:
            return AgentResult(self.name, topic, error=f"{type(exc).__name__}: {exc}")


class SearchResearchAgent:
    """Coordinator adapter for the Playwright-based SearchAgent."""

    name = SearchAgent.name

    def __init__(self, result_count: int = 10):
        self.result_count = result_count

    async def research(self, topic: str, headless: bool = False) -> AgentResult:
        try:
            result = await SearchAgent(result_count=self.result_count).research(topic, headless=headless)
            return AgentResult(self.name, topic, result=result)
        except Exception as exc:
            return AgentResult(self.name, topic, error=f"{type(exc).__name__}: {exc}")


class Coordinator:
    """Dispatch topics to agents and combine their results."""

    def __init__(self, agents: Optional[List[ResearchAgent]] = None, result_count: int = 10):
        self.agents = agents or [SearchResearchAgent(result_count=result_count)]

    async def run(self, topic: str, headless: bool = False) -> List[AgentResult]:
        """Run all registered agents for a topic.

        ``gather`` allows independent agents to work concurrently when more
        agents are added. The browser agent currently opens its own browser.
        """
        print(f"Coordinating {len(self.agents)} agent(s) for: {topic}")
        results = await asyncio.gather(
            *(agent.research(topic, headless=headless) for agent in self.agents)
        )
        return list(results)

    @staticmethod
    def report(results: List[AgentResult]) -> None:
        print("\n=== Research results ===")
        for result in results:
            print(f"\n[{result.agent_name}]")
            if result.error:
                print(f"Error: {result.error}")
            elif result.result is None:
                print("No result returned.")
            elif isinstance(result.result, dict):
                for key, value in result.result.items():
                    if value:
                        print(f"{key}: {value}")
            elif isinstance(result.result, list) and result.result and hasattr(result.result[0], "url"):
                for ranking in result.result:
                    print(f"{ranking.url} — {ranking.relevance} relevant, {ranking.trustworthiness} trustworthy")
            else:
                print(result.result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coordinate computer-use research agents")
    parser.add_argument("--headless", action="store_true", help="run browsers without showing them")
    parser.add_argument("--count", type=int, default=10, help="number of URLs to find and rank (default: 10)")
    parser.add_argument("topic", nargs="?", help="research topic; otherwise you will be prompted")
    return parser.parse_args()


async def async_main(topic: str, count: int = 10, headless: bool = False) -> List[AgentResult]:
    search_result = await SearchResearchAgent(result_count=count).research(topic, headless=headless)
    if search_result.error:
        results = [search_result]
    else:
        ranking_result = await RankingResearchAgent(search_result.result).research(topic, headless=headless)
        results = [search_result, ranking_result]
    Coordinator.report(results)
    return results


def main() -> None:
    args = parse_args()
    topic = args.topic or input("What topic would you like to research? ").strip()
    if not topic:
        raise SystemExit("A non-empty research topic is required.")
    if args.count < 1:
        raise SystemExit("--count must be at least 1.")
    asyncio.run(async_main(topic, count=args.count, headless=args.headless))


if __name__ == "__main__":
    main()

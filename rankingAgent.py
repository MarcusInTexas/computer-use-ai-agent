"""Rank researched web pages for relevance and trustworthiness.

RankingAgent uses Playwright to read each URL's rendered content, then sends a
bounded excerpt to an LLM reasoner. Configure the reasoner with:

    OPENAI_API_KEY=...
    OPENAI_MODEL=gpt-4o-mini                 # optional
    OPENAI_BASE_URL=https://api.openai.com/v1 # optional

The endpoint is OpenAI-compatible, so it can also point at a compatible local
or hosted provider. Scores are always clamped to 0..100 and parsed as JSON.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from playwright.async_api import Browser, Page, async_playwright

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
MAX_PAGE_CHARS = 12_000


@dataclass
class Ranking:
    url: str
    relevance: int
    trustworthiness: int
    explanation: str = ""


class LLMReasoner:
    """Small OpenAI-compatible JSON reasoner with no extra SDK dependency."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for RankingAgent")

    def rank(self, topic: str, url: str, page_text: str) -> Ranking:
        prompt = f"""Evaluate this web page for research on the topic below.

Topic: {topic}
URL: {url}
Page content excerpt:
{page_text[:MAX_PAGE_CHARS]}

Return ONLY valid JSON with exactly these keys:
{{"relevance": <integer 0-100>, "trustworthiness": <integer 0-100>, "explanation": "brief reason"}}
Relevance measures how useful the page is for the topic. Trustworthiness considers
source authority, evidence, transparency, and whether the page appears reliable.
Do not infer trustworthiness solely from a high search ranking or from the URL.
"""
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a careful web-source evaluator. Follow the requested JSON format."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        body = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt >= self.max_retries:
                    if exc.code == 429:
                        raise RuntimeError("LLM request was rate-limited (HTTP 429). Try again later or use a project with available API quota.") from exc
                    raise RuntimeError(f"LLM request failed: HTTP {exc.code}") from exc
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = max(1.0, float(retry_after)) if retry_after else 2 ** attempt
                except ValueError:
                    delay = 2 ** attempt
                print(f"LLM rate limit reached; retrying in {delay:g}s ({attempt + 1}/{self.max_retries})")
                time.sleep(delay)
            except urllib.error.URLError as exc:
                raise RuntimeError(f"LLM request failed: {exc}") from exc
        if body is None:
            raise RuntimeError("LLM request returned no response")
        content = body["choices"][0]["message"]["content"]
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        try:
            result = json.loads(content)
            relevance = self._score(result.get("relevance"))
            trust = self._score(result.get("trustworthiness"))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM returned invalid ranking JSON: {content}") from exc
        return Ranking(url, relevance, trust, str(result.get("explanation", "")))

    @staticmethod
    def _score(value: Any) -> int:
        score = int(round(float(value)))
        return max(0, min(100, score))


class RankingAgent:
    name = "ranking-agent"

    def __init__(self, reasoner: Optional[LLMReasoner] = None):
        self.reasoner = reasoner or LLMReasoner()

    async def read_page(self, page: Page, url: str) -> str:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(500)
            return await page.locator("body").inner_text(timeout=10_000)
        except Exception as exc:
            return f"Unable to read page: {type(exc).__name__}: {exc}"

    async def rank_urls(self, topic: str, urls: List[str], headless: bool = False) -> List[Ranking]:
        if not urls:
            raise ValueError("RankingAgent received no URLs")
        async with async_playwright() as playwright:
            browser: Browser = await playwright.chromium.launch(headless=headless)
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
            rankings: List[Ranking] = []
            for url in urls:
                text = await self.read_page(page, url)
                ranking = await asyncio.to_thread(self.reasoner.rank, topic, url, text)
                rankings.append(ranking)
                print(f"RankingAgent rates {url} {ranking.relevance} relevant and {ranking.trustworthiness} trustworthy")
            await browser.close()
        return rankings

    async def research(self, topic: str, urls: List[str], headless: bool = False) -> List[Ranking]:
        return await self.rank_urls(topic, urls, headless=headless)


async def async_main(topic: str, urls: List[str], headless: bool = False) -> List[Ranking]:
    return await RankingAgent().research(topic, urls, headless=headless)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank web pages with an LLM")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("topic")
    parser.add_argument("urls", nargs="+", help="URLs to evaluate")
    args = parser.parse_args()
    asyncio.run(async_main(args.topic, args.urls, headless=args.headless))


if __name__ == "__main__":
    main()

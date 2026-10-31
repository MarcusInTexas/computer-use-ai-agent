"""Bing-based web search agent with LLM title relevance ranking.

SearchAgent uses Playwright to search Bing, collects result names and URLs,
and asks an LLM to rank relevance using *only the result names*. It never
sends the referenced pages' contents to this relevance reasoner.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, List, Optional
from urllib.parse import parse_qs, quote_plus, urlparse

from playwright.async_api import Browser, Page, async_playwright

SEARCH_URL = "https://www.bing.com/search?q={}"
DEFAULT_RESULT_COUNT = 10


@dataclass
class SearchResult:
    name: str
    url: str


class TitleRelevanceReasoner:
    """LLM reasoner that receives topic and titles, never page contents."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for SearchAgent title ranking")

    def rank(self, topic: str, results: List[SearchResult]) -> List[SearchResult]:
        numbered = "\n".join(f"{i}: {result.name}" for i, result in enumerate(results))
        prompt = f"""Rank these search-result names for relevance to the research topic.
Topic: {topic}

Search-result names (and nothing else):
{numbered}

Return ONLY JSON in this exact form: {{"indices": [<indices from most relevant to least relevant>]}}.
Include every index exactly once. Judge relevance from the names only; do not infer or
request the contents of the linked pages. The first {len(results)} indices should be returned."""
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You rank search result titles carefully and return valid JSON."},
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
                    raise RuntimeError(f"LLM title-ranking request failed: HTTP {exc.code}") from exc
                delay = 2 ** attempt
                print(f"LLM rate limit reached; retrying in {delay}s ({attempt + 1}/{self.max_retries})")
                time.sleep(delay)
            except urllib.error.URLError as exc:
                raise RuntimeError(f"LLM title-ranking request failed: {exc}") from exc
        if body is None:
            raise RuntimeError("LLM title-ranking request returned no response")
        content = body["choices"][0]["message"]["content"]
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        try:
            indices = json.loads(content)["indices"]
            order = [int(index) for index in indices]
            if sorted(order) != list(range(len(results))):
                raise ValueError("indices were not a complete permutation")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM returned invalid title ranking JSON: {content}") from exc
        return [results[index] for index in order]


class SearchAgent:
    name = "search-agent"

    def __init__(self, result_count: int = DEFAULT_RESULT_COUNT, max_pages: int = 5,
                 reasoner: Optional[TitleRelevanceReasoner] = None):
        if result_count < 1:
            raise ValueError("result_count must be at least 1")
        self.result_count = result_count
        self.max_pages = max_pages
        self.reasoner = reasoner or TitleRelevanceReasoner()

    async def observe(self, page: Page) -> List[SearchResult]:
        """Observe Bing's rendered result titles and links.

        Bing's markup varies between sessions, so inspect rendered h2 elements
        and find their anchors in JavaScript instead of relying on one CSS
        container class.
        """
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_timeout(1500)
        raw_results = await page.locator("h2").evaluate_all(
            """els => els.map(h2 => {
                const a = h2.querySelector('a');
                return a ? {name: (a.textContent || '').trim(), url: a.href} : null;
            }).filter(Boolean)"""
        )
        return [
            SearchResult(name=item["name"], url=item["url"])
            for item in raw_results
            if item.get("name") and item.get("url")
        ]

    @staticmethod
    def _valid_results(results: List[SearchResult]) -> List[SearchResult]:
        seen = set()
        valid = []
        for result in results:
            parsed = urlparse(result.url)
            url = result.url.split("#", 1)[0]
            if parsed.netloc in {"bing.com", "www.bing.com"} and parsed.path.startswith("/ck/a"):
                encoded = parse_qs(parsed.query).get("u", [""])[0]
                if encoded.startswith("a1"):
                    try:
                        url = base64.urlsafe_b64decode(encoded[2:] + "===").decode("utf-8")
                        parsed = urlparse(url)
                    except (ValueError, UnicodeDecodeError):
                        continue
            if parsed.scheme in {"http", "https"} and parsed.netloc not in {"bing.com", "www.bing.com"} and url not in seen:
                seen.add(url)
                valid.append(SearchResult(result.name, url))
        return valid

    async def search(self, topic: str, headless: bool = False) -> List[str]:
        if not topic.strip():
            raise ValueError("topic must not be empty")
        async with async_playwright() as playwright:
            browser: Browser = await playwright.chromium.launch(headless=headless)
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
            await page.goto(SEARCH_URL.format(quote_plus(topic)), wait_until="domcontentloaded")
            candidates: List[SearchResult] = []
            for step in range(1, self.max_pages + 1):
                candidates.extend(await self.observe(page))
                candidates = self._valid_results(candidates)
                print(f"[search step {step}] observed {len(candidates)} Bing results")
                if len(candidates) >= self.result_count:
                    break
                next_link = page.locator("a#pnnext").first
                if not await next_link.count():
                    break
                await next_link.click()
                await page.wait_for_load_state("domcontentloaded")
            await browser.close()
        if not candidates:
            raise RuntimeError("Bing returned no usable search results")
        ranked = await asyncio.to_thread(self.reasoner.rank, topic, candidates)
        selected = ranked[: self.result_count]
        print("\nSearchAgent has found:")
        for result in selected:
            print(f"{result.name} — {result.url}")
        return [result.url for result in selected]

    async def research(self, topic: str, headless: bool = False) -> List[str]:
        return await self.search(topic, headless=headless)


async def async_main(topic: str, count: int, headless: bool = False) -> List[str]:
    return await SearchAgent(result_count=count).research(topic, headless=headless)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Bing and rank result names with an LLM")
    parser.add_argument("--count", type=int, default=DEFAULT_RESULT_COUNT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("topic", nargs="?")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")
    topic = args.topic or input("What topic would you like to research? ").strip()
    if not topic:
        parser.error("a non-empty topic is required")
    asyncio.run(async_main(topic, count=args.count, headless=args.headless))


if __name__ == "__main__":
    main()

import itertools
import re

import utils
from logger import logger


def extract(params: dict) -> list[str]:
    if type(params) != dict:
        return []

    urls = params.get("urls", [])
    if not urls or type(urls) != list:
        logger.error(f"[AALIB] skip extract due to cannot found any valid task")
        return []

    try:
        results = utils.multi_thread_collect(extract_one, urls)
        links = list(set(list(itertools.chain.from_iterable(results))))
        logger.info(f"[AALIB] crawl finished, found {len(links)} sites: {links}")
        return links
    except:
        return []


def extract_one(url: str) -> list[str]:
    if not utils.isurl(url=url):
        logger.error(f"[AALIB] skip execute extract because url: {url} is invalid")
        return []

    content = utils.http_get(url=url)
    if not content:
        logger.error(
            f"[AALIB] cannot obtain any site due to no response for url: {url}"
        )
        return []

    try:
        groups = re.findall(r"<tr.*>([\s\S]*?)</tr>", content)
        if not groups:
            logger.warn(f"[AALIB] cannot found any domains from [{url}]")
            return []

        candidates = []
        for group in groups:
            if "导航" in group or "可用" not in group:
                continue

            matchers = re.findall(
                r"<a href.*target=\"_blank\">([\s\w\-:/\.]+)</a>", group
            )
            if not matchers:
                continue

            for domain in matchers:
                site = utils.url_complete(domain.strip().lower())
                if site:
                    candidates.append(site)

        logger.info(
            f"[AALIB] extract {url} finished, found {len(candidates)} sites: {candidates}"
        )
        return candidates
    except:
        logger.error(f"[AALIB] occur error when extract domains from [{url}]")
        return []

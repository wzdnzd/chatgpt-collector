import re

import utils
from logger import logger


def extract(params: dict) -> list[str]:
    if type(params) != dict:
        return []

    url = params.get("url", "https://link.lbbai.com")
    content = utils.http_get(url=url)
    if not content:
        return []

    try:
        groups = re.findall("<tr.*>([\s\S]*?)</tr>", content)
        if not groups:
            logger.warn(f"[LBBAIWarn] cannot found any domains from [{url}]")
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
            f"[LBBAIInfo] extract finished, found {len(candidates)} sites: {candidates}"
        )
        return candidates
    except:
        logger.error(f"[LBBAIError] occur error when extract domains from [{url}]")
        return []

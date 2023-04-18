import re

import utils
from logger import logger


def fetch(params: dict) -> list[str]:
    if type(params) != dict:
        return []

    url = params.get(
        "url", "https://raw.githubusercontent.com/xx025/carrot/main/README.md"
    )
    content = utils.http_get(url=url)
    if not content:
        return []

    try:
        groups = re.findall("<tr>([\s\S]*?)</tr>", content)
        if not groups:
            logger.warn(f"[CarrotWarn] cannot found any domains from [{url}]")
            return []

        candidates = []
        for group in groups:
            if params.get("temporary", True):
                flag = ("😄" in group and "🔑" not in group) or (
                    "🎁" in group and not re.search(r"注册|登录", group)
                )
            else:
                flag = "😄" in group and "🔑" not in group

            if not flag:
                continue

            matchers = re.findall(
                r"<td><a href.*target=\"_blank\">([\s\w\-:/\.]+)</a>(?:\s+)?</td>",
                group,
            )
            if not matchers:
                continue

            for domain in matchers:
                site = domain.strip().lower()
                if site:
                    candidates.append(f"https://{site}")

        # failure sites
        if params.get("lapse", False):
            groups = re.findall(r"\d+\.(?:\s+)?(https://.*)(?:\s+)?<br/>", content)
            for group in groups:
                site = utils.extract_domain(
                    group.strip().lower(), include_protocal=True
                )
                if site:
                    candidates.append(site)

        logger.info(
            f"[CarrotInfo] extract finished, found {len(candidates)} sites: {candidates}"
        )
        return candidates
    except:
        logger.error(f"[CarrotError] occur error when extract domains from [{url}]")
        return []

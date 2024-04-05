# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2022-07-19

import json
import re

import utils
from logger import logger

FILTER_KEYS = set(
    [
        "invalid",
        "hide",
        "needKey",
        "needPwd",
        "needLogin",
        "needPay",
        "needVerify",
        "errmsg",
    ]
)


def fetch(params: dict) -> list[str]:
    if type(params) != dict:
        return []

    # see: https://github.com/lzwme/chatgpt-sites
    url = "https://raw.githubusercontent.com/lzwme/chatgpt-sites/main/site-info.json"

    exclude, sites = params.get("exclude", {}), set()
    if not exclude or type(exclude) != dict:
        exclude = {}

    site_exclude = utils.trim(exclude.get("site", ""))
    repo_exclude = utils.trim(exclude.get("repo", ""))
    title_exclude = utils.trim(exclude.get("title", ""))

    try:
        content = utils.http_get(url=url)
        if content:
            data = json.loads(content).get("siteInfo", {})
            for u, v in data.items():
                if not u or not v or type(v) != dict:
                    continue

                # v.keys cannot contains any key which in FILTER_KEYS
                if not FILTER_KEYS.isdisjoint(set(v.keys())):
                    continue

                try:
                    if site_exclude and re.search(site_exclude, u, flags=re.I):
                        continue
                    repo, title = v.get("repo", ""), v.get("title", "")
                    if (repo and repo_exclude and re.search(repo_exclude, repo, flags=re.I)) or (
                        title and title_exclude and re.search(title_exclude, title, flags=re.I)
                    ):
                        continue

                    sites.add(u)
                except:
                    logger.error(f"[LZWME] invalid exclude regex")
    except:
        logger.error(f"[LZWME] extract error due to exception")

    logger.info(f"[LZWME] crawl finished, found {len(sites)} sites")
    return list(sites)

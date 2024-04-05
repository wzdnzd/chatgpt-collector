# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2022-07-19

import re
from urllib import parse

import urlvalidator
import utils
from logger import logger


def extract(params: dict) -> list[str]:
    if type(params) != dict:
        return []

    url = utils.trim(
        params.get(
            "url",
            "https://liaobots.site/_next/static/chunks/pages/index-675bcb5ed5a7e0c0.js",
        )
    )
    if not urlvalidator.isurl(url=url):
        logger.error(f"[LiaoBots] skip extract because url is invalid")
        return []

    sites, exclude = set(), utils.trim(params.get("exclude", ""))
    urlpath = utils.trim(params.get("urlpath", ""))
    mode = utils.trim(params.get("mode", ""))
    try:
        content = utils.http_get(url=url)
        if content:
            groups = re.findall(
                'u="(https?://)",(?:\s+)?f="(\S+)",(?:\s+)?h=.*?,(?:\s+)?m=\{(.*?)\};',
                content,
                flags=re.I,
            )
            for group in groups:
                if len(group) != 3:
                    continue

                domains = re.findall('"(.*?)"', group[2], flags=re.I)
                for domain in domains:
                    site = f"{group[0]}{group[1]}{domain}"
                    if exclude and re.search(exclude, site, flags=re.I):
                        continue

                    if urlpath:
                        site = parse.urljoin(site, urlpath)
                    if mode:
                        site += f"?mode={mode}"

                    sites.add(site)
    except:
        logger.error(f"[LiaoBots] extract error due to exception")

    logger.info(f"[LiaoBots] crawl finished, found {len(sites)} sites")
    return list(sites)

# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2022-07-18

import itertools
import json
import math
import re

import utils
from aes import AESCipher
from logger import logger


def collect(params: dict) -> list[str]:
    params = {} if type(params) != dict else params
    url = utils.trim(params.get("url", ""))
    url = "https://chatgpt-site.zhaoyeqing.cn" if not url else url

    passphrase = utils.trim(params.get("passphrase", ""))
    prefix = "https://api.zhaoyeqing.cn"

    content, subpath = utils.http_get(url=url), ""
    if not content:
        logger.warn(f"[ZYQ] failed to fetch content from url: {url}")
    else:
        # <script defer="defer" src="/static/js/main.ad39aa59.js">
        groups = re.findall(r'<script\s+.*?src="(/static/js/main.[a-z0-9]+.js)">', content, flags=re.I)
        if groups:
            subpath = groups[0]

    if subpath:
        url += subpath
        content = utils.http_get(url=url)
        if content:
            regex = r"AES\.decrypt\(e\.web_url,\"(.*?)\"\)\.toString"
            match = re.findall(regex, content, flags=re.I)
            passphrase = match[0] if match else passphrase

            pattern = r"baseURL:\"(https://.*?)\""
            group = re.findall(pattern, content, flags=re.I)
            domain = utils.trim(group[0]) if group else ""
            prefix = domain if domain else prefix
        else:
            logger.warn(f"[ZYQ] query api address and decrypy key failed from main.js")

    if utils.isblank(passphrase):
        logger.error(f"[ZYQ] cannot decrypt due to passphrase is empty")
        return []

    # https://api.zhaoyeqing.cn/chatgpt/page
    url = f"{prefix}/chatgpt/page"
    peer = max(params.get("peer", 24), 1)

    types = params.get("types", [])
    include = utils.trim(params.get("include", ""))
    exclude = utils.trim(params.get("exclude", ""))
    sitetypes = set(types) if types else None

    # fetch data
    links, content = partition(url=url, peer=peer)
    sites = []
    if content:
        sites = request_once(
            url="",
            passphrase=passphrase,
            include=include,
            exclude=exclude,
            sitetypes=sitetypes,
            content=content,
        )
    if links:
        tasks = [[x, passphrase, include, exclude, sitetypes, ""] for x in links]
        results = utils.multi_thread_collect(func=request_once, tasks=tasks)
        sites = list(set(list(itertools.chain.from_iterable(results))))

    logger.info(f"[ZYQ] crawl finished, found {len(sites)} sites")
    return sites


def partition(url: str, peer: int) -> tuple[list, str]:
    peer, links = max(peer, 1), None
    content = utils.http_get(url=f"{url}/1/{peer}")
    if content:
        try:
            response = json.loads(content)
            if not response.get("data", []):
                content = ""
            else:
                total = response.get("allTotal", peer)
                times = math.ceil(total / peer)
                if times > 1:
                    content = ""
                    links = [f"{url}/{x}/{peer}" for x in range(1, times + 1)]
        except:
            logger.error(f"[ZYQ] parse response data error for total sites count")
            content = ""

    return links, content


def request_once(
    url: str,
    passphrase: str,
    include: str = "",
    exclude: str = "",
    sitetypes: set = None,
    content: str = "",
) -> list:
    if (not url and not content) or not passphrase:
        return []

    if not content:
        content = utils.http_get(url=url)
        if not content:
            logger.error(f"[ZYQ] fetch websites json data error, url: {url}")
            return []

    sites = set()
    try:
        data = json.loads(content).get("data", [])
        cipher = AESCipher(passphrase=passphrase)
        for item in data:
            if not item or (sitetypes and item.get("cn_key", "") not in sitetypes):
                continue

            website = item.get("web_url", "")
            try:
                url = utils.url_complete(site=cipher.decrypt(website))
                if not url:
                    continue

                if include and not re.search(include, url, re.I):
                    continue
                else:
                    if exclude and re.search(exclude, url, re.I):
                        continue

                sites.add(url)
            except:
                logger.warn(f"[ZYQ] invalid web url: {website} or include: {include} or exclude: {exclude} is invalid")
    except:
        logger.error(f"[ZYQ] occue error when parse data for url: {url}")

    return list(sites)

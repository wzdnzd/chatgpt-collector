import itertools
import re

import urlvalidator
import utils
from logger import logger


def fetch(params: dict) -> list[str]:
    if type(params) != dict:
        return []

    tasks = []
    for k, v in params.items():
        if not k or not v or type(v) != dict or not v.pop("enable", True):
            continue
        tasks.append([k, v.get("options", {})])

    if not tasks:
        logger.error(f"[Carrot] skip extract due to cannot found any valid task")
        return []

    try:
        results = utils.multi_thread_collect(crawlone, tasks)
        links = list(set(list(itertools.chain.from_iterable(results))))
        logger.info(f"[Carrot] crawl finished, found {len(links)} sites: {links}")
        return links
    except Exception as e:
        logger.error(f"[Carrot] error when colletct: {str(e)}")
        return []


def crawlone(url: str, params: dict) -> list[str]:
    if not urlvalidator.isurl(url=url) or not params or type(params) != dict:
        logger.error(f"[Carrot] invalid task configuration, url: {url} or params: {params} is empty")
        return []

    content, candidates = utils.http_get(url=url), []
    try:
        groups = re.findall(r"<tr>([\s\S]*?)</tr>", content)
        if not groups:
            logger.warn(f"[Carrot] cannot found any domains from [{url}]")
            return []

        includes = set(params.get("include", []))
        excludes = set(params.get("exclude", []))
        regex = utils.trim(params.get("regex", ""))
        if not regex:
            regex = r"<td><a href.*target=\"_blank\">([\s\w\-:/\.]+)</a>(?:\s+)?</td>"

        for group in groups:
            if re.search(r"注册|登录", group):
                continue
            fi, fe = False, False
            for label in includes:
                fi = label in group
                if fi:
                    break

            # not in includes
            if not fi:
                continue

            for label in excludes:
                fe = label in group
                if fe:
                    break

            # in excludes
            if fe:
                continue

            matchers = re.findall(regex, group)
            if not matchers:
                continue

            for domain in matchers:
                site = utils.url_complete(domain.strip().lower())
                if site:
                    candidates.append(site)

        # failure sites
        if params.get("lapse", False):
            groups = re.findall(r"\d+\.(?:\s+)?(https://.*)(?:\s+)?<br/>", content)
            for group in groups:
                site = utils.url_complete(utils.extract_domain(group.strip().lower(), include_protocal=True))
                if site:
                    candidates.append(site)

        logger.info(f"[Carrot] extract url: {url} finished, found {len(candidates)} sites: {candidates}")
    except:
        logger.error(f"[Carrot] occur error when extract domains from [{url}]")

    return candidates

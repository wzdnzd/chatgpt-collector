# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2023-06-30

import itertools
import json
import math
import os
import re
import time
from datetime import datetime, timedelta

import push
import utils
from logger import logger

# date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# last modified key name
LAST_MODIFIED = "lastModified"

# github rest api prefix
GITHUB_API = "https://api.github.com"

# github username
OWNER = "Yidadaa"

# github repository name
REPO = "ChatGPT-Next-Web"

# request headers
DEFAULT_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": utils.USER_AGENT,
}

# github token
GITHUB_TOKEN = utils.trim(os.environ.get("GITHUB_TOKEN", ""))
if GITHUB_TOKEN:
    DEFAULT_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def last_history(url: str) -> datetime:
    last = datetime(year=1970, month=1, day=1)
    content = utils.http_get(url=url) if url else ""
    if content:
        modified = ""
        try:
            modified = json.loads(content).get(LAST_MODIFIED, "")
            if modified:
                last = datetime.strptime(modified, DATE_FORMAT) + timedelta(minutes=-10)
        except Exception:
            logger.error(f"[NextWeb] invalid date format: {modified}")

    return last


def query_forks_count(retry: int = 3) -> int:
    url = f"{GITHUB_API}/search/repositories?q=user:{OWNER}+repo:{REPO}+{REPO}"
    content = utils.http_get(url=url, headers=DEFAULT_HEADERS, retry=retry)
    if not content:
        logger.error(f"[NextWeb] failed to query forks count")
        return -1

    try:
        items = json.loads(content).get("items", [])
        if not items or type(items) != list:
            logger.error(f"[NextWeb] cannot get forks count, message: {content}")
            return -1

        return items[0].get("forks_count", 0)
    except:
        logger.error(
            f"[NextWeb] occur error when parse forks count, message: {content}"
        )
        return -1


def list_deployments(history: str) -> list[str]:
    last = last_history(url=history)
    count, peer = query_forks_count(retry=3), 100
    total = int(math.ceil(count / peer))

    # concurrent
    if last.year == 1970:
        tasks = [[x, last, peer] for x in range(1, total + 1)]
        results = utils.multi_thread_collect(func=query_deployments_page, params=tasks)
        return list(itertools.chain.from_iterable([x[0] for x in results]))

    # serial
    deployments, over, page = [], False, 1
    while not over and page <= total:
        array, over = query_deployments_page(page=page, last=last, peer=peer)
        deployments.extend(array)
        page += 1

    return deployments


def parse_site(
    url: str, page: int = 1, peer: int = 5, session: str = "", rest: bool = True
) -> str:
    if not rest and session:
        return extract_target(url=url, session=session)

    if not url or page <= 0:
        return ""

    peer, domain = min(max(1, peer), 100), ""
    url = f"{url}?environment=production&per_page={peer}&page={page}"
    content = utils.http_get(url=url, headers=DEFAULT_HEADERS, interval=2.0)
    if not content:
        return ""

    try:
        deployments, index = json.loads(content), 0
        while not domain and index < len(deployments):
            item = deployments[index]
            if item and type(item) == dict:
                domain = query_deployment_status(url=item.get("statuses_url", ""))

            index += 1
    except:
        logger.error(
            f"[NextWeb] failed to parse target url due to cannot query deployments, message: {content}"
        )

    return domain


def extract_target(url: str, session: str) -> str:
    if not url or not session:
        return ""

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": utils.USER_AGENT,
        "Cookie": f"user_session={session}",
    }

    url = url.replace(f"{GITHUB_API}/repos", "https://github.com")
    content = utils.http_get(url=url, headers=headers, interval=1.0)
    regex = '<a class="btn btn-outline".*? href="(https://.*?)">View deployment</a>'
    matcher = re.search(regex, content, flags=re.I)
    return matcher.group(1) if matcher else ""


def query_deployment_status(url: str) -> str:
    if not url:
        return ""

    content = utils.http_get(url=url, headers=DEFAULT_HEADERS, interval=1.0)
    try:
        statuses = json.loads(content)
        if not statuses or type(statuses) != list:
            return ""

        state = statuses[0].get("state", "")
        target = statuses[0].get("target_url", "")
        return target if state == "success" else ""
    except:
        return ""


def query_deployments_page(
    page: int, last: datetime, peer: int = 100
) -> tuple[list[str], bool]:
    if page <= 0 or not last:
        return [], False

    peer = min(max(peer, 1), 100)
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/forks?sort=newest&per_page={peer}&page={page}"
    deployments, over, starttime = list(), False, time.time()

    content = utils.http_get(url=url, headers=DEFAULT_HEADERS, interval=1.0)
    try:
        forks = json.loads(content)
        for fork in forks:
            if not fork or type(fork) != dict:
                continue

            datetext = fork.get("created_at", "")
            updated_at = datetime.strptime(datetext, "%Y-%m-%dT%H:%M:%SZ")

            # last time has already been collected
            if updated_at <= last:
                over = True
                break

            deployment = fork.get("deployments_url", "")
            if deployment:
                deployments.append(deployment)
    except:
        logger.error(
            f"[NextWeb] cannot fetch deployments for page: {page}, message: {content}"
        )

    cost = "{:.2f}s".format(time.time() - starttime)
    logger.info(f"[NextWeb] finished query deployments for page: {page}, cost: {cost}")

    return deployments, over


def check(url: str) -> str:
    if not url:
        return ""

    response = utils.http_post_noerror(
        url=f"{url}/api/config", allow_redirects=False, retry=2
    )
    if not response or response.getcode() != 200:
        return ""

    try:
        content = response.read().decode("UTF8")
        data = json.loads(content)
        return "" if data.get("needCode", False) else url
    except:
        return ""


def collect(params: dict) -> list:
    if not params or type(params) != dict:
        return []

    persist = params.get("persist", {})
    server = os.environ.get("SUBSCRIBE_CONF", "").strip()
    pushtool = push.get_instance(domain=server)
    if (
        not persist
        or type(persist) != dict
        or not pushtool.validate(persist.get("modified", {}))
    ):
        logger.error(f"[NextWeb] invalid persist config, please check it and try again")
        return []

    # last modified storage config
    store, sites = persist.get("modified", {}), []
    # github user session
    session = utils.trim(os.environ.get("USER_SESSION", ""))

    begin = datetime.utcnow().strftime(DATE_FORMAT)
    deployments = list_deployments(history=pushtool.raw_url(push_conf=store))
    if deployments:
        args = [[x, 1, 5, session, True] for x in deployments]
        collections = utils.multi_thread_collect(func=parse_site, params=args)
        candidates = utils.multi_thread_collect(func=check, params=collections)
        sites = [x for x in candidates if x]

    # save last modified time
    content = json.dumps({LAST_MODIFIED: begin})
    pushtool.push_to(content=content, push_conf=store, group="modified")

    logger.info(f"[NextWeb] crawl finished, found {len(sites)} sites")
    return sites

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

# deployments filepath
DEPLOYMENTS_FILE = os.path.join(utils.PATH, "data", "deployments.txt")

# domains filepath
MATERIAL_FILE = os.path.join(utils.PATH, "data", "material.txt")

# candidates filepath
CANDIDATES_FILE = os.path.join(utils.PATH, "data", "candidates.txt")

# result filepath
SITES_FILE = os.path.join(utils.PATH, "data", "sites.txt")

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

# running on local
LOCAL_MODE = utils.trim(os.environ.get("LOCAL_MODE", "")).lower() in ["true", "1"]


def last_history(url: str, refresh: bool) -> datetime:
    last = datetime(year=1970, month=1, day=1)
    if not refresh and url:
        content = utils.http_get(url=url) if url else ""
        if content:
            modified = ""
            try:
                modified = json.loads(content).get(LAST_MODIFIED, "")
                if modified:
                    last = datetime.strptime(modified, DATE_FORMAT) + timedelta(
                        minutes=-10
                    )
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


def list_deployments(history: str, sort: str = "newest", refresh=False) -> list[str]:
    last = last_history(history, refresh)
    count, peer = query_forks_count(retry=3), 100
    total = int(math.ceil(count / peer))

    # see: https://docs.github.com/en/rest/repos/forks?apiVersion=2022-11-28
    if sort not in ["newest", "oldest", "stargazers", "watchers"]:
        sort = "newest"

    # concurrent
    if last.year == 1970:
        tasks = [[x, last, peer, sort] for x in range(1, total + 1)]
        results = utils.multi_thread_collect(func=query_deployments_page, params=tasks)
        return list(itertools.chain.from_iterable([x[0] for x in results]))

    # serial
    deployments, over, page = [], False, 1
    while not over and page <= total:
        array, over = query_deployments_page(page=page, last=last, peer=peer, sort=sort)
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

    if LOCAL_MODE and domain:
        utils.write_file(filename=MATERIAL_FILE, lines=domain, overwrite=False)

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
    page: int, last: datetime, peer: int = 100, sort: str = "newest"
) -> tuple[list[str], bool]:
    if page <= 0 or not last:
        return [], False

    peer = min(max(peer, 1), 100)
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/forks?sort={sort}&per_page={peer}&page={page}"
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


def auth(url: str) -> str:
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
        target = "" if data.get("needCode", False) else url

        if LOCAL_MODE and target:
            filename = os.path.join(utils.PATH, "data", CANDIDATES_FILE)
            utils.write_file(filename=filename, lines=target, overwrite=False)

        return target
    except:
        return ""


def check(domain: str) -> str:
    if not domain:
        return ""

    url = f"{domain}/api/openai/dashboard/billing/subscription"
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/json",
        "Referer": url,
        "User-Agent": utils.USER_AGENT,
    }

    content = utils.http_get(url=url, headers=headers, interval=0.5, retry=2)
    try:
        limit = json.loads(content).get("hard_limit_usd", "")
        target = (
            f"{domain}/api/openai/v1/chat/completions?mode=openai&stream=true"
            if utils.is_number(limit)
            else ""
        )

        if LOCAL_MODE and target:
            utils.write_file(filename=SITES_FILE, lines=target, overwrite=False)

        return target
    except:
        return ""


def read(filepath: str) -> list[str]:
    filepath, collections = utils.trim(filepath), []

    # if file not exist, read from it's backup
    for file in [filepath, f"{filepath}.bak"]:
        if not file or not os.path.exists(file) or not os.path.isfile(file):
            continue

        with open(file, "r", encoding="utf8") as f:
            for line in f.readlines():
                collections.append(line.replace("\n", "").strip().lower())
        break

    return collections


def load(url: str, overlay: bool) -> list[str]:
    url, sites = utils.trim(url), []

    # load local file if exist
    sites.extend(read(SITES_FILE))

    # add local existing material if necessary
    if overlay:
        sites.extend(read(MATERIAL_FILE))

    # load remote sites
    if url:
        content = utils.http_get(url=url)
        sites.extend(content.split(","))

    return [utils.extract_domain(url=x, include_protocal=True) for x in sites]


def backup_file(filepath: str) -> None:
    if not filepath or not os.path.exists(filepath) or not os.path.isfile(filepath):
        return

    newfile = f"{filepath}.bak"
    if os.path.exists(newfile):
        os.remove(newfile)

    os.rename(filepath, newfile)


def collect(params: dict) -> list:
    if not params or type(params) != dict:
        return []

    persist = params.get("persist", {})
    storage = {} if not persist or type(persist) != dict else persist

    server = os.environ.get("SUBSCRIBE_CONF", "").strip()
    pushtool = push.get_instance(domain=server)
    if not LOCAL_MODE and not pushtool.validate(storage.get("modified", {})):
        logger.error(
            f"[NextWeb] invalid persist config, must config modified store if running on remote"
        )
        return []

    # store config
    modified, database = storage.get("modified", {}), storage.get("sites", {})

    # github user session
    session = utils.trim(os.environ.get("USER_SESSION", ""))

    # sort type for forks
    sort = utils.trim(params.get("sort", "newest")).lower()

    # check only
    checkonly = params.get("checkonly", False)

    # re-generate deployments
    refresh = False if checkonly else params.get("refresh", False)

    # add local existing material if necessary
    overlay = params.get("overlay", False)

    mode, starttime = "LOCAL" if LOCAL_MODE else "", time.time()
    logger.info(
        f"[NextWeb] start to collect sites from {OWNER}/{REPO}, mode: {mode}, checkonly: {checkonly}, refresh: {refresh}"
    )

    # load exists
    candidates = [] if refresh else load(pushtool.raw_url(database), overlay)

    if not checkonly:
        begin = datetime.utcnow().strftime(DATE_FORMAT)

        # TODO: if it is possible to bypass the rate limiting measures of GitHub, asynchronous requests can be used
        deployments = list_deployments(
            pushtool.raw_url(push_conf=modified), sort, refresh
        )

        # add local existing deployments if necessary
        if overlay:
            deployments.extend(read(filepath=DEPLOYMENTS_FILE))

        # deduplication
        deployments = list(set(deployments))
        logger.info(
            f"[NextWeb] collect completed, found {len(deployments)} deployments"
        )

        if deployments:
            # save deployments to file
            if LOCAL_MODE:
                # backup exist files
                for filename in [DEPLOYMENTS_FILE, MATERIAL_FILE, CANDIDATES_FILE]:
                    backup_file(filepath=filename)

                utils.write_file(DEPLOYMENTS_FILE, deployments, True)

            args = [[x, 1, 5, session, True] for x in deployments]
            candidates.extend(utils.multi_thread_collect(func=parse_site, params=args))

        # save last modified time
        if pushtool.validate(modified):
            content = json.dumps({LAST_MODIFIED: begin})
            pushtool.push_to(content=content, push_conf=modified, group="modified")

    # backup sites.txt if file exists
    backup_file(filepath=SITES_FILE)

    # concurrent check
    results = utils.multi_thread_collect(func=check, params=list(set(candidates)))
    sites = [x for x in results if x]

    # save sites
    if sites and pushtool.validate(database):
        success = pushtool.push_to(
            content=",".join(sites), push_conf=database, group="sites"
        )
        if not success:
            logger.error(f"[NextWeb] push {len(sites)} sites to remote failed")

    cost = "{:.2f}s".format(time.time() - starttime)
    logger.info(
        f"[NextWeb] finished check {len(results)} candidates, got {len(sites)} avaiable sites, cost: {cost}"
    )
    return sites

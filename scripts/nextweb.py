# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2023-06-30

import asyncio
import itertools
import json
import math
import os
import re
import time
import warnings
from datetime import datetime, timedelta, timezone
from urllib import parse as parse

import interactive
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
OWNER = "ChatGPTNextWeb"

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

# running on local
LOCAL_MODE = utils.trim(os.environ.get("LOCAL_MODE", "")).lower() in ["true", "1"]


def last_history(url: str, refresh: bool) -> datetime:
    last = datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)
    if not refresh and url:
        content = utils.http_get(url=url) if url else ""
        if content:
            modified = ""
            try:
                modified = json.loads(content).get(LAST_MODIFIED, "")
                if modified:
                    date = datetime.strptime(modified, DATE_FORMAT)
                    last = (date + timedelta(minutes=-10)).replace(tzinfo=timezone.utc)
            except Exception:
                logger.error(f"[NextWeb] invalid date format: {modified}")

    return last


def generate_path(repository: str, filename: str, username: str = "") -> str:
    filename = utils.trim(filename)
    if not filename:
        raise ValueError("filename cannot be empty")

    repository = utils.trim(repository).lower()
    username = utils.trim(username).lower()

    return os.path.join(utils.PATH, "data", username, repository, filename)


def query_forks_count(username: str, repository: str, retry: int = 3) -> int:
    username = utils.trim(username)
    repository = utils.trim(repository)
    if not username or not repository:
        logger.error(f"[NextWeb] invalid github username or repository")
        return -1

    url = f"{GITHUB_API}/search/repositories?q=user:{username}+repo:{repository}+{repository}"
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
        logger.error(f"[NextWeb] occur error when parse forks count, message: {content}")
        return -1


def list_deployments(username: str, repository: str, history: datetime, sort: str = "newest") -> list[str]:
    username = utils.trim(username)
    repository = utils.trim(repository)
    if not username or not repository:
        logger.error(f"[NextWeb] cannot list deployments from github due to username or repository is empty")
        return []

    last = history or datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)
    count, peer = query_forks_count(username=username, repository=repository, retry=3), 100
    total = int(math.ceil(count / peer))

    # see: https://docs.github.com/en/rest/repos/forks?apiVersion=2022-11-28
    if sort not in ["newest", "oldest", "stargazers", "watchers"]:
        sort = "newest"

    # concurrent
    if last.year == 1970:
        tasks = [[username, repository, x, last, peer, sort] for x in range(1, total + 1)]
        results = utils.multi_thread_run(func=query_deployments_page, tasks=tasks)
        return list(itertools.chain.from_iterable([x[0] for x in results]))

    # serial
    deployments, over, page = [], False, 1
    while not over and page <= total:
        array, over = query_deployments_page(
            username=username,
            repository=repository,
            page=page,
            last=last,
            peer=peer,
            sort=sort,
        )

        deployments.extend(array)
        page += 1

    return deployments


def parse_site(
    url: str,
    filename: str,
    page: int = 1,
    peer: int = 5,
    session: str = "",
    rest: bool = True,
    last: datetime = None,
    exclude: str = "",
) -> list[str]:
    if not rest and session:
        return extract_target(url=url, session=session, exclude=exclude)

    if not url or page <= 0:
        return []

    filename = utils.trim(filename)
    if not filename:
        logger.error(f"you must specify a filename to save parse result")
        return []

    peer, domains = min(max(1, peer), 100), set()

    # fetch homepage
    homepage = extract_homepage(url=url)
    if homepage:
        domains.add(homepage)

    if not last:
        last = datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)

    url = f"{url}?environment=production&per_page={peer}&page={page}"
    content = utils.http_get(url=url, headers=DEFAULT_HEADERS, interval=2.0)

    if content:
        try:
            deployments = json.loads(content)
            for item in deployments:
                if not item or not isinstance(item, dict):
                    continue

                # get deployment updat time
                text = utils.trim(item.get("updated_at", ""))

                # updated = datetime.fromisoformat(text.replace("Z", "+00:00")) if text else None
                updated = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) if text else None

                # it has been processed last time
                if updated and updated <= last:
                    continue

                site = query_deployment_status(url=item.get("statuses_url", ""), exclude=exclude)
                if site:
                    domains.add(site)
        except:
            logger.error(f"[NextWeb] failed to parse target url due to cannot query deployments, message: {content}")

    targets = [] if not domains else list(domains)

    if LOCAL_MODE and targets:
        utils.write_file(filename=filename, lines=targets, overwrite=False)

    return targets


def extract_homepage(url: str) -> str:
    url = utils.trim(url).removesuffix("/deployments")
    if not url:
        return ""

    content = utils.http_get(url=url, headers=DEFAULT_HEADERS, interval=2.0)
    if not content:
        return ""

    try:
        data = json.loads(content)
        if not data:
            return ""

        homepage = data.get("homepage", "")
        if homepage and homepage.startswith("https://github.com/"):
            homepage = ""

        return homepage
    except:
        return ""


def extract_target(url: str, session: str, exclude: str = "") -> list[str]:
    if not url or not session:
        return []

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-Agent": utils.USER_AGENT,
        "Cookie": f"user_session={session}",
    }

    url = url.replace(f"{GITHUB_API}/repos", "https://github.com")
    content = utils.http_get(url=url, headers=headers, interval=1.0)
    regex = '<a href="(https://.*?)" data-testid="deployments-list-environment-url" tabindex="-1">'
    groups = re.findall(regex, content, flags=re.I)
    exclude = utils.trim(exclude)

    if not exclude or not groups:
        return [] if not groups else groups

    domains = list()
    for domain in groups:
        try:
            if re.search(exclude, domain, flags=re.I) is not None:
                continue
        except:
            logger.warning(f"[NextWeb] invalid exclude regex: {exclude}")

        domains.append(domain)

    return domains


def query_deployment_status(url: str, exclude: str = "") -> str:
    if not url:
        return ""

    content = utils.http_get(url=url, headers=DEFAULT_HEADERS, interval=1.0)
    try:
        statuses = json.loads(content)
        if not statuses or type(statuses) != list:
            return ""

        state = statuses[0].get("state", "")
        target = statuses[0].get("environment_url", "") or ""

        if target == "https://" or target == "http://":
            return ""

        exclude = utils.trim(exclude)
        if exclude:
            try:
                if re.search(exclude, target, flags=re.I) is not None:
                    return ""
            except:
                logger.warning(f"[NextWeb] invalid exclude regex: {exclude}")

        success = state == "success"
        if not success:
            description = utils.trim(statuses[0].get("description", ""))
            success = re.search("checks for deployment have failed", description, flags=re.I) is not None

        return target if success else ""
    except:
        return ""


def query_deployments_page(
    username: str,
    repository: str,
    page: int,
    last: datetime,
    peer: int = 100,
    sort: str = "newest",
) -> tuple[list[str], bool]:
    username = utils.trim(username)
    repository = utils.trim(repository)

    if not username or not repository or page <= 0 or not last:
        return [], False

    peer = min(max(peer, 1), 100)
    url = f"{GITHUB_API}/repos/{username}/{repository}/forks?sort={sort}&per_page={peer}&page={page}"
    deployments, over, starttime = list(), False, time.time()

    content, retry = "", 5
    while not content and retry > 0:
        content = utils.http_get(url=url, headers=DEFAULT_HEADERS, interval=1.0)
        retry -= 1
        if not content:
            time.sleep(2)

    try:
        forks = json.loads(content)
        for fork in forks:
            if not fork or type(fork) != dict:
                continue

            datetext = fork.get("created_at", "")
            updated = datetime.strptime(datetext, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

            # last time has already been collected
            if updated <= last:
                over = True
                break

            deployment = fork.get("deployments_url", "")
            if deployment:
                deployments.append(deployment)
    except:
        logger.error(f"[NextWeb] cannot fetch deployments for page: {page}, message: {content}")

    cost = "{:.2f}s".format(time.time() - starttime)
    logger.info(f"[NextWeb] finished query deployments for page: {page}, cost: {cost}")

    return deployments, over


def auth(url: str, filename: str) -> str:
    filename = utils.trim(filename)
    if not url or not filename:
        return ""

    response = utils.http_post_noerror(url=f"{url}/api/config", allow_redirects=False, retry=2)
    if not response or response.getcode() != 200:
        return ""

    try:
        content = response.read().decode("UTF8")
        data = json.loads(content)
        target = "" if data.get("needCode", False) else url

        if LOCAL_MODE and target:
            utils.write_file(filename=filename, lines=target, overwrite=False)

        return target
    except:
        return ""


def check_billing(domain: str) -> bool:
    warnings.warn("This API has been disabled upstream", DeprecationWarning)

    if not domain:
        return False

    url = f"{domain}/api/openai/dashboard/billing/subscription"
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/json",
        "Referer": url,
        "Origin": url,
        "User-Agent": utils.USER_AGENT,
    }

    content = utils.http_get(url=url, headers=headers, interval=0.5, retry=2)
    try:
        limit = json.loads(content).get("hard_limit_usd", "")
        return utils.is_number(limit)
    except:
        return False


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


def load(url: str, overlay: bool, sites_file: str = "", material_file: str = "") -> list[str]:
    url, sites = utils.trim(url), []

    # load local file if exist
    sites.extend(read(filepath=sites_file))

    # add local existing material if necessary
    if overlay:
        sites.extend(read(filepath=material_file))

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

    server = os.environ.get("COLLECT_CONF", "").strip()
    pushtool = push.get_instance(domain=server)
    if not LOCAL_MODE and not pushtool.validate(storage.get("modified", {})):
        logger.error(f"[NextWeb] invalid persist config, must config modified store if running on remote")
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

    # number of concurrent threads
    num_threads = params.get("num_threads", 0)

    # run type
    run_async = params.get("async", True)

    # github username
    username = utils.trim(params.get("username", "")) or OWNER

    # github repository
    repository = utils.trim(params.get("repository", "")) or REPO

    # deployments filepath
    deployments_file = generate_path(repository=repository, filename="deployments.txt")

    # domains filepath
    material_file = generate_path(repository=repository, filename="material.txt")

    # candidates filepath
    candidates_file = generate_path(repository=repository, filename="candidates.txt")

    # result filepath
    sites_file = generate_path(repository=repository, filename="sites.txt")

    mode, starttime = "LOCAL" if LOCAL_MODE else "REMOTE", time.time()
    logger.info(
        f"[NextWeb] start to collect sites from {username}/{repository}, mode: {mode}, checkonly: {checkonly}, refresh: {refresh}"
    )

    # load exists
    candidates = [] if refresh else load(pushtool.raw_url(database), overlay, sites_file, material_file)

    if not checkonly:
        begin = datetime.now(timezone.utc).strftime(DATE_FORMAT)

        # fetch last run time
        last = last_history(pushtool.raw_url(push_conf=modified), refresh)

        # TODO: if it is possible to bypass the rate limiting measures of GitHub, asynchronous requests can be used
        deployments = list_deployments(username, repository, last, sort)

        # add local existing deployments if necessary
        if overlay:
            deployments.extend(read(filepath=deployments_file))

        # deduplication
        deployments = list(set(deployments))
        logger.info(f"[NextWeb] collect completed, found {len(deployments)} deployments")

        if deployments:
            # save deployments to file
            if LOCAL_MODE:
                # backup exist files
                for file in [deployments_file, material_file, candidates_file]:
                    backup_file(filepath=file)

                utils.write_file(deployments_file, deployments, True)

            # regex for dropped domains
            exclude = utils.trim(params.get("exclude", ""))

            args = [[x, material_file, 1, 100, session, True, last, exclude] for x in deployments]
            logger.info(f"[NextWeb] extract target domain begin, count: {len(args)}")

            materials = utils.multi_thread_run(
                func=parse_site,
                tasks=args,
                show_progress=True,
                num_threads=num_threads,
            )
            newsites = list(itertools.chain.from_iterable(materials))
            candidates.extend([x for x in newsites if x])

        # save last modified time
        if pushtool.validate(modified):
            content = json.dumps({LAST_MODIFIED: begin})
            pushtool.push_to(content=content, push_conf=modified, group="modified")

    logger.info(f"[NextWeb] candidate target collection completed, found {len(candidates)} sites")

    if params.get("skip_check", False) or not candidates:
        logger.warning("[NextWeb] availability testing steps will be skipped")
        return list(set(candidates))

    # backup sites.txt if file exists
    backup_file(filepath=sites_file)

    candidates = list(set(candidates))
    chunk = max(params.get("chunk", 256), 1)
    model = params.get("model", "") or "gpt-3.5-turbo"
    standard = params.get("standard", False)
    filename = sites_file if LOCAL_MODE else ""

    logger.info(f"[NextWeb] start to check available, sites: {len(candidates)}, model: {model}")

    # batch check
    sites = interactive.batch_probe(
        candidates=candidates,
        model=model,
        filename=filename,
        standard=standard,
        run_async=run_async,
        show_progress=True,
        num_threads=num_threads,
        chunk=chunk,
    )

    # save sites
    if sites and pushtool.validate(database):
        success = pushtool.push_to(content=",".join(sites), push_conf=database, group="sites")
        if not success:
            logger.error(f"[NextWeb] push {len(sites)} sites to remote failed")

    cost = "{:.2f}s".format(time.time() - starttime)
    logger.info(f"[NextWeb] finished check {len(candidates)} candidates, got {len(sites)} avaiable sites, cost: {cost}")
    return sites

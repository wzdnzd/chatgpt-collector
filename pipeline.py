# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2023-06-30

import itertools
import json
import math
import os
import re
import time
import warnings
from datetime import datetime, timedelta, timezone
from functools import cache
from urllib import parse as parse

import interactive
import push
import utils
from logger import logger
from urlvalidator import isurl

# date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# last modified key name
LAST_MODIFIED = "lastModified"

# github rest api prefix
GITHUB_API = "https://api.github.com"

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


@cache
def is_local() -> bool:
    return utils.trim(os.environ.get("LOCAL_MODE", "")).lower() in ["true", "1"]


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
                logger.error(f"[Pipeline] invalid date format: {modified}")

    return last


def generate_path(repository: str, filename: str, username: str = "", concat: bool = False) -> str:
    filename = utils.trim(filename)
    if not filename:
        raise ValueError("filename cannot be empty")

    repository = utils.trim(repository).lower()
    username = utils.trim(username).lower()
    subpath = repository if not concat else f"{username}-{repository}"

    return os.path.join(utils.PATH, "data", subpath, filename)


def query_forks_count(username: str, repository: str, retry: int = 3) -> int:
    username = utils.trim(username)
    repository = utils.trim(repository)
    if not username or not repository:
        logger.error(f"[Pipeline] invalid github username or repository")
        return -1

    url = f"{GITHUB_API}/repos/{username}/{repository}"
    content = utils.http_get(url=url, headers=DEFAULT_HEADERS, retry=retry, interval=1)
    if not content:
        logger.error(f"[Pipeline] failed to query forks count")
        return -1

    try:
        data = json.loads(content)
        return data.get("forks_count", 0)
    except:
        logger.error(f"[Pipeline] occur error when parse forks count, message: {content}")
        return -1


def list_deployments(username: str, repository: str, history: datetime, sort: str = "newest") -> list[tuple[str, str]]:
    username = utils.trim(username)
    repository = utils.trim(repository)
    if not username or not repository:
        logger.error(f"[Pipeline] cannot list deployments from github due to username or repository is empty")
        return []

    last = history or datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)
    count, peer = query_forks_count(username=username, repository=repository, retry=3), 100
    total = int(math.ceil(count / peer))

    logger.info(f"[Pipeline] query forks finished, forks: {count}, page: {total}")

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
    homepage: bool = False,
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
    if homepage:
        website = extract_homepage(url=url)
        if website:
            domains.add(website)

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
            logger.error(f"[Pipeline] failed to parse target url due to cannot query deployments, message: {content}")

    targets = [] if not domains else list(domains)

    if is_local() and targets:
        utils.write_file(filename=filename, lines=targets, overwrite=False)

    return targets


def real_deployment(url: str) -> str:
    def rsplit_once(text: str, patterns: str) -> tuple[str, str, str]:
        pattern = "|".join(f"(?:{p})" for p in patterns)
        matches = list(re.finditer(pattern, text))

        if not matches:
            return text, "", ""

        match = matches[-1]
        word = match.group(0)
        start, end = match.span()

        return text[:start], text[end:], word

    url = utils.trim(url)
    if not isurl(url):
        return ""

    if not url.endswith(".vercel.app"):
        return url

    patterns = ["-projects", "-team"]
    left, right, keyword = rsplit_once(url, patterns)
    if not keyword or not right:
        return url

    prefix, project, _ = rsplit_once(left, ["-[A-Za-z0-9]{9}-"])
    target = f"{prefix}-{project}{keyword}{right}"

    return target


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

        homepage = utils.extract_domain(url=data.get("homepage", ""), include_protocal=True).lower()
        if homepage and (not isurl(homepage) or homepage.startswith("https://github.com/")):
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
            logger.warning(f"[Pipeline] invalid exclude regex: {exclude}")

        domains.append(real_deployment(url=domain))

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
                logger.warning(f"[Pipeline] invalid exclude regex: {exclude}")

        success = state == "success"
        if not success:
            description = utils.trim(statuses[0].get("description", ""))
            success = re.search("checks for deployment have failed", description, flags=re.I) is not None

        return real_deployment(url=target) if success else ""
    except:
        return ""


def query_deployments_page(
    username: str,
    repository: str,
    page: int,
    last: datetime,
    peer: int = 100,
    sort: str = "newest",
) -> tuple[list[tuple[str, str]], bool]:
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

            datetext = fork.get("updated_at", "")
            updated = datetime.strptime(datetext, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

            # last time has already been collected
            if updated <= last:
                over = True
                break

            deployment = utils.trim(fork.get("deployments_url", "")).lower()
            homepage = utils.extract_domain(url=utils.trim(fork.get("homepage", "")), include_protocal=True).lower()
            if not isurl(homepage) or homepage.startswith("https://github.com/"):
                homepage = ""

            if deployment or homepage:
                deployments.append((deployment, homepage))
    except:
        logger.error(f"[Pipeline] cannot fetch deployments for page: {page}, message: {content}")

    cost = "{:.2f}s".format(time.time() - starttime)
    logger.info(f"[Pipeline] finished query deployments for page: {page}, cost: {cost}")

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

        if is_local() and target:
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


def tidy(filepath: str, together: bool = True) -> None:
    filepath = utils.trim(filepath)
    if not filepath or not os.path.exists(filepath) or not os.path.isfile(filepath):
        return

    lines = None
    try:
        with open(filepath, "r", encoding="utf8") as f:
            lines = list(set([x.replace("\n", "").strip().lower() for x in f.readlines() if x]))
    except:
        logger.error(f"[Pipeline] skip tidy due to read file {filepath} failed")
        return

    if not lines:
        return

    # put the same URL together
    if together:
        lines.sort(key=lambda x: x[::-1])

    text = "\n".join(lines)

    # write to file
    try:
        with open(filepath, "w+", encoding="utf8") as f:
            f.write(text)
            f.flush()
    except:
        logger.error(f"[Pipeline] skip tidy due to write file {filepath} failed")


def collect(params: dict) -> list:
    if not params or type(params) != dict:
        return []

    storage = params.get("storage", {})
    persist = {} if not storage or type(storage) != dict else storage.get("items", {})
    if not persist or type(persist) != dict:
        persist = dict()

    pushtool = push.get_instance(push_config=push.PushConfig.from_dict(storage))
    if not is_local() and not pushtool.validate(persist.get("modified", {})):
        logger.error(f"[Pipeline] invalid persist config, must config modified store if running on remote")
        return []

    # store config
    modified, database = persist.get("modified", {}), persist.get("sites", {})

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

    # regex for dropped domains
    exclude = utils.trim(params.get("exclude", ""))

    # request body template
    style = params.get("style", 0)

    # customize request headers
    headers = params.get("headers", "")

    # github username
    username = utils.trim(params.get("username", ""))

    # github repository
    repository = utils.trim(params.get("repository", ""))

    # whether to concatenate username and repository as the folder name
    concat = params.get("concat", False)

    if not username or not repository:
        logger.error(f"[Pipeline] github username or repository cannot be blank")
        return []

    # deployments filepath
    deployments_file = generate_path(
        username=username, repository=repository, filename="deployments.txt", concat=concat
    )

    # domains filepath
    material_file = generate_path(username=username, repository=repository, filename="material.txt", concat=concat)

    # candidates filepath
    candidates_file = generate_path(username=username, repository=repository, filename="candidates.txt", concat=concat)

    # result filepath
    sites_file = generate_path(username=username, repository=repository, filename="sites.txt", concat=concat)

    mode, starttime = "LOCAL" if is_local() else "REMOTE", time.time()
    logger.info(
        f"[Pipeline] start to collect sites from {username}/{repository}, mode: {mode}, checkonly: {checkonly}, refresh: {refresh}"
    )

    # load exists
    candidates = [] if refresh else load(pushtool.raw_url(database), overlay, sites_file, material_file)

    if not checkonly:
        begin = datetime.now(timezone.utc).strftime(DATE_FORMAT)

        # fetch last run time
        last = last_history(pushtool.raw_url(push_conf=modified), refresh)

        # source repository deployments
        domains, deployments = set(), [f"https://api.github.com/repos/{username}/{repository}/deployments"]

        # TODO: if it is possible to bypass the rate limiting measures of GitHub, asynchronous requests can be used
        websites = list_deployments(username, repository, last, sort)
        if websites:
            for deployment, homepage in websites:
                if deployment:
                    deployments.append(deployment)
                if homepage:
                    try:
                        if re.search(exclude, homepage, flags=re.I) is not None:
                            continue
                    except:
                        logger.warning(f"[Pipeline] invalid exclude regex: {exclude}")

                    domains.add(homepage)

        # add local existing deployments if necessary
        if overlay:
            deployments.extend(read(filepath=deployments_file))

        # deduplication
        deployments = list(set(deployments))
        logger.info(f"[Pipeline] collect completed, found {len(deployments)} deployments")

        if deployments or domains:
            homepages = list(domains) if domains else []
            candidates.extend(homepages)

            # save deployments to file
            if is_local():
                # backup exist files
                for file in [deployments_file, material_file, candidates_file]:
                    utils.backup_file(filepath=file)

                if deployments:
                    utils.write_file(deployments_file, deployments, True)
                if homepages:
                    utils.write_file(material_file, homepages, True)

            if deployments:
                args = [[x, material_file, 1, 100, session, True, last, exclude, False] for x in deployments]
                logger.info(f"[Pipeline] extract target domain begin, count: {len(args)}")

                materials = utils.multi_thread_run(
                    func=parse_site,
                    tasks=args,
                    show_progress=True,
                    num_threads=num_threads,
                )
                newsites = [] if not materials else list(itertools.chain.from_iterable(materials))
                candidates.extend([x for x in newsites if x])

                # deduplication
                tidy(filepath=material_file, together=True)

        # save last modified time
        if pushtool.validate(modified):
            content = json.dumps({LAST_MODIFIED: begin})
            pushtool.push_to(content=content, push_conf=modified, group="modified")

    logger.info(f"[Pipeline] candidate target collection completed, found {len(candidates)} sites")

    if params.get("skip_check", False) or not candidates:
        logger.warning("[Pipeline] availability testing steps will be skipped")
        return list(set(candidates))

    # backup sites.txt if file exists
    utils.backup_file(filepath=sites_file)

    candidates = list(set(candidates))
    chunk = max(params.get("chunk", 256), 1)
    model = params.get("model", "") or "gpt-3.5-turbo"
    potentials = params.get("potentials", "")
    wander = params.get("wander", False)
    strict = params.get("strict", False)
    filename = sites_file if is_local() else ""

    logger.info(f"[Pipeline] start to check available, sites: {len(candidates)}, model: {model}")

    # batch check
    sites = interactive.batch_probe(
        candidates=candidates,
        question=params.get("question", ""),
        keyword=params.get("keyword", ""),
        model=model,
        filename=filename,
        potentials=potentials,
        wander=wander,
        run_async=run_async,
        show_progress=True,
        num_threads=num_threads,
        chunk=chunk,
        style=style,
        headers=headers,
        strict=strict,
    )

    # deduplication and sort
    tidy(filepath=filename, together=True)

    # save sites
    if sites and pushtool.validate(database):
        success = pushtool.push_to(content=",".join(sites), push_conf=database, group="sites")
        if not success:
            logger.error(f"[Pipeline] push {len(sites)} sites to remote failed")

    cost = "{:.2f}s".format(time.time() - starttime)
    logger.info(
        f"[Pipeline] finished check {len(candidates)} candidates, got {len(sites)} avaiable sites, cost: {cost}"
    )
    return sites

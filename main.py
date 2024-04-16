# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2022-07-15

import argparse
import importlib
import itertools
import json
import multiprocessing
import os
import re
import time
import traceback
from multiprocessing.managers import ListProxy
from multiprocessing.synchronize import Semaphore
from urllib import parse

import interactive
import push
import urlvalidator
import utils
from logger import logger
from urlvalidator import isurl

# available group name
AVAILABLES = "availables"

# candidate group name
CANDIDATES = "candidates"


def execute_script(script: str, params: dict = {}) -> list[str]:
    try:
        # format: a.b.c#function or a-b.c#_function or a#function and so on
        regex = r"^([a-zA-Z0-9_]+|([0-9a-zA-Z_]+([a-zA-Z0-9_\-]+)?\.)+)[a-zA-Z0-9_\-]+#[a-zA-Z_]+[0-9a-zA-Z_]+$"
        if not re.match(regex, script):
            logger.info(f"[ScriptError] script execute error because script: {script} is invalidate")
            return []

        path, func_name = script.split("#", maxsplit=1)
        path = f"scripts.{path}"
        module = importlib.import_module(path)
        if not hasattr(module, func_name):
            logger.error(f"script: {path} not exists function {func_name}")
            return []

        func = getattr(module, func_name)

        starttime = time.time()
        logger.info(f"[ScriptInfo] start execute script: scripts.{script}")

        sites = func(params)
        if type(sites) != list:
            logger.error(f"[ScriptError] return value error, need a list, but got a {type(sites)}")
            return []

        endtime = time.time()
        logger.info(
            "[ScriptInfo] finished execute script: scripts.{}, cost: {:.2f}s".format(script, endtime - starttime)
        )

        return sites
    except:
        logger.error(f"[ScriptError] occur error run script: {script}, message: \n{traceback.format_exc()}")
        return []


def call(script: str, params: dict, availables: ListProxy, semaphore: Semaphore) -> None:
    try:
        if not script:
            return

        subscribes = execute_script(script=script, params=params)
        if subscribes and type(subscribes) == list:
            availables.extend(subscribes)
    finally:
        if semaphore is not None and isinstance(semaphore, Semaphore):
            semaphore.release()


def batch_call(tasks: dict) -> list[str]:
    if not tasks:
        return []

    try:
        thread_num = max(min(len(tasks), 50), 1)
        with multiprocessing.Manager() as manager:
            availables, processes = manager.list(), []
            semaphore = multiprocessing.Semaphore(thread_num)
            for k, v in tasks.items():
                semaphore.acquire()
                p = multiprocessing.Process(target=call, args=(k, v, availables, semaphore))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            return list(availables)
    except:
        traceback.print_exc()
        return []


def crawl_pages(tasks: dict) -> list[str]:
    if not tasks or type(tasks) != dict:
        return []

    params = []
    for k, v in tasks.items():
        if not k or type(v) != dict:
            continue

        include = v.get("include", "").strip()
        exclude = v.get("exclude", "").strip()
        regex = v.get("regex", "").strip()
        urlpath = v.get("urlpath", "").strip()
        mode = v.get("mode", "").strip()
        params.append([k, include, exclude, regex, urlpath, mode])

    if not params:
        return []

    try:
        results = utils.multi_thread_run(func=crawl_single_page, tasks=params)
        links = list(set(list(itertools.chain.from_iterable(results))))

        logger.info(f"[PagesCrawl] crawl from pages finished, found {len(links)} sites")
        return links
    except:
        traceback.print_exc()
        return []


def crawl_single_page(
    url: str,
    include: str,
    exclude: str = "",
    regex: str = "",
    uripath: str = "",
    mode: str = "",
) -> list[str]:
    if not utils.isurl(url=url) or (utils.isblank(include) and utils.isblank(regex)):
        logger.error(f"[PageError] invalid task configuration, must specify url and include or regex")
        return []

    try:
        content, collections = utils.http_get(url=url), set()
        if content == "":
            logger.error(f"[PageError] invalid response, site: {url}")
            return []

        regex = (
            r"https?://(?:[a-zA-Z0-9\u4e00-\u9fa5\-]+\.)+[a-zA-Z0-9\u4e00-\u9fa5\-]+" if utils.isblank(regex) else regex
        )
        groups = re.findall(regex, content, flags=re.I)

        for item in groups:
            try:
                if not re.search(include, item, flags=re.I) or (exclude and re.search(exclude, item, flags=re.I)):
                    continue

                item = utils.url_complete(item)
                if item:
                    collections.add(item)
            except:
                logger.error(
                    f"[PageError] maybe pattern 'include' or 'exclude' exists some problems, include: {include}\texclude: {exclude}"
                )

        sites = list(collections)
        logger.info(f"[PageInfo] crawl page {url} finished, found {len(sites)} sites: {sites}")

        uripath, mode = utils.trim(uripath).lower(), utils.trim(mode).lower()

        # regular url path
        if not re.match(r"^(\/?\w+)+((\.)?\w+|\/)(\?(\w+=[\w\d]+(&\w+=[\w\d]+)*)+){0,1}$", uripath):
            uripath = ""
        mode = "" if not mode else parse.urlencode({"mode": mode})
        if not uripath and not mode:
            return sites

        collections = []
        for site in sites:
            if uripath:
                site = parse.urljoin(base=site, url=uripath)
            if mode:
                symbol = "&" if "?" in site else "?"
                site = f"{site}{symbol}{mode}"

            collections.append(site)

        return collections
    except:
        logger.error(f"[PageError] occur error when crawl site=[{url}]")
        return []


def process(args: argparse.Namespace) -> None:
    url = args.config
    if not url or not isurl(url):
        logger.error("[ConfigError] cannot process because config is invalid")
        return
    try:
        headers = {"User-Agent": utils.USER_AGENT, "Referer": url}
        content = utils.http_get(url=url, headers=headers)
        if not content:
            logger.error(f"[ConfigError] cannot fetch configuration from url: {url}")
            return
        else:
            configuration = json.loads(content)
            pushtool = push.get_instance(domain=url)
            tasks = regularized(data=configuration, pushtool=pushtool)

            if not tasks:
                logger.warning("[ConfigError] cannot found any legal collect task from scripts and pages")
                return

            urls = batch_call(tasks.get("scripts", {}))
            urls.extend(crawl_pages(tasks.get("pages", {})))

            # exist urls
            candidates = fetct_exist(tasks.get("persists"), pushtool)

            # generate blacklist
            blacklist = generate_blacklist(
                persists=tasks.get("persists"),
                blackconf=tasks.get("blacklist", {}),
                pushtool=pushtool,
            )

            # strictly conforms to the event-stream standard
            strict = tasks.get("stream", {}).get("strict", True)

            data = detect(
                urls=urls,
                candidates=candidates,
                blacklist=blacklist,
                model=args.model,
                filename=args.filename,
                run_async=args.run_async,
                num_threads=args.num,
                show=args.show,
                strict=strict,
                threshold=tasks.get("threshold"),
            )

            if not data:
                logger.warning("[ProcessWarn] cannot found any domains")
                return

            # all persist config
            persists = tasks.get("persists", {})
            if not persists or not isinstance(persists, dict):
                persists = dict()

            params = [[v, persists.get(k), k, 5] for k, v in data.items() if v]
            cleaned = os.environ.get("NO_PARAMS", "false") in ["true", "1"]

            # support event-stream sites
            isolation = tasks.get("stream", {}).get("isolation", False)
            name = tasks.get("stream", {}).get("persist", "")
            streaming = persists.get(name, {})

            if isolation and streaming:
                sites = data.get(AVAILABLES, "")
                fastly = [x.split("?")[0] if cleaned else x for x in sites.split(",") if "stream=true" in x]

                logger.info(f"[ProcessInfo] collected {len(fastly)} faster sites that support event-stream")
                text = ",".join(fastly)
                if text:
                    params.append([text, streaming, name, 5])

            # save by model version classification
            versioned = tasks.get("versioned", {})
            if not versioned or not isinstance(versioned, dict):
                versioned = {}

            if versioned.get("enable", True):
                for i in interactive.support_version():
                    key, flag = f"gpt{i}", f"version={i}"
                    if key not in versioned:
                        continue

                    group = persists.get(versioned.get(key), {})
                    if not group:
                        continue

                    items = [
                        x.split("?")[0] if cleaned else remove_url_param(url=x, key="version")
                        for x in sites.split(",")
                        if flag in x
                    ]

                    logger.info(f"[ProcessInfo] collected {len(items)} sites for group {key}")

                    text = ",".join(items)
                    if text:
                        params.append([text, group, key, 5])

            cpu_count = multiprocessing.cpu_count()
            num = len(params) if len(params) <= cpu_count else cpu_count

            pool = multiprocessing.Pool(num)
            pool.starmap(func=pushtool.push_to, iterable=params)
            pool.close()
    except:
        traceback.print_exc()
        logger.error("[ConfigError] occur error when load task config")


def regularized(data: dict, pushtool: push.PushTo) -> dict:
    if not data or not pushtool:
        return {}

    persists, groups = data.pop("persists", {}), {}
    if not persists or type(persists) != dict:
        logger.error("[ConfigError] collect configuration must specify persists")
        return {}

    for k, v in persists.items():
        if pushtool.validate(v):
            groups[k] = v

    if CANDIDATES not in groups or AVAILABLES not in groups:
        logger.error("[ConfigError] persists must include 'availables' and 'candidates'")
        return {}

    scripts, pages = data.pop("scripts", {}), data.pop("pages", {})
    script_tasks, page_tasks = {}, {}

    if scripts and type(scripts) == dict:
        for k, v in scripts.items():
            if utils.isblank(k) or type(v) != dict or not v.get("enable", True):
                continue
            script_tasks[k] = v.get("params", {})

    if pages and type(pages) == dict:
        for k, v in pages.items():
            if not utils.isurl(k) or type(v) != dict or not v.pop("enable", True):
                continue

            page_tasks[k] = v.get("params", {})

    threshold = max(int(data.get("threshold", 72)), 1)
    data.update(
        {
            "threshold": threshold,
            "scripts": script_tasks,
            "pages": page_tasks,
            "persists": groups,
        }
    )
    return data


def fetct_exist(persists: dict, pushtool: push.PushTo) -> dict:
    if not persists or not pushtool:
        return {}

    candidates = {}
    for k, v in persists.items():
        if k != AVAILABLES and k != CANDIDATES:
            continue

        content = utils.http_get(url=pushtool.raw_url(v))
        if not content:
            continue

        if k == AVAILABLES:
            for site in content.split(","):
                if not utils.isblank(site):
                    candidates[site] = {"defeat": 0}
        elif k == CANDIDATES:
            try:
                candidates.update(json.loads(content))
            except:
                logger.error("[CandidatesError] fetch candidates failed")

    return candidates


def generate_blacklist(persists: dict, blackconf: dict, pushtool: push.PushTo) -> dict:
    disable = os.environ.get("DISABLE_BLACKLIST", "") in ["true", "1"]
    if disable or not persists or not blackconf:
        return {}

    address, autoadd = blackconf.get("address", ""), blackconf.get("auto", False)
    if address not in persists or address in [AVAILABLES, CANDIDATES]:
        return {}

    url = pushtool.raw_url(persists.get(address))
    content, regex = utils.http_get(url=url), ""
    if content:
        sites = list(set([x for x in content.split("\n") if not utils.isblank(x) and not x.startswith("#")]))
        regex = "|".join(sites)

    return {"persist": address, "auto": autoadd, "regex": regex}


def merge_blacklist(old: str, sites: list[str]) -> str:
    if not sites:
        return old
    if not old:
        return "\n".join(sites)

    uniquesites = set(old.split("|"))
    uniquesites.update(sites)

    return "\n".join(list(uniquesites))


def intercept(sites: list[str], blacklist: str = "") -> list[str]:
    if not sites:
        return []

    dataset, pattern = {}, None
    if blacklist:
        try:
            pattern = re.compile(blacklist, flags=re.I)
        except:
            logger.error(f"[InterceptError] blacklist=[{blacklist}] is invalid")

    for site in sites:
        if not urlvalidator.isurl(site) or (pattern and pattern.search(site)):
            continue

        hostname = parse.urlparse(url=site).netloc
        url = dataset.get(hostname, "")
        if not url or len(url) < len(site):
            dataset[hostname] = site

    return list(dataset.values())


def detect(urls: list[str], candidates: dict, blacklist: dict = {}, **kwargs) -> dict:
    if not urls and not candidates:
        return {}

    urls = urls if urls is not None else []
    candidates = candidates if candidates is not None else {}
    urls.extend(candidates.keys())

    # automatically add sites whose failure times exceed the threshold to the blacklist
    blacklist = {} if not isinstance(blacklist, dict) else blacklist
    auto_addblack = blacklist.get("auto", False)
    regex = blacklist.get("regex", "")

    sites = intercept(sites=urls, blacklist=regex)
    starttime = time.time()
    logger.info(f"[ProbeSites] starting sites probe, count: {len(sites)}")

    items = interactive.batch_probe(
        candidates=sites,
        model=kwargs.get("model", ""),
        filename=kwargs.get("filename", ""),
        standard=False,
        run_async=kwargs.get("run_async", True),
        show_progress=kwargs.get("show", False),
        num_threads=kwargs.get("num_threads", 0),
    )

    threshold = max(1, kwargs.get("threshold", 7))
    potentials, blacksites = {}, []
    availables = parse_domains(items)
    overalls = parse_domains(sites)

    for k, v in overalls.items():
        if not k or not v or k in availables or v not in candidates:
            continue

        defeat = candidates.get(v).get("defeat", 0) + 1 if v in candidates else 1
        if defeat <= threshold:
            potentials[v] = {"defeat": defeat}
        elif auto_addblack:
            blacksites.append(k)

    cost = "{:.2f}s".format(time.time() - starttime)
    logger.info(f"[ProbeInfo] collect finished, cost: {cost}, found {len(items)} sites")

    if auto_addblack:
        logger.warning(f"[ProbeWarn] add {len(blacksites)} sites to blacklist: {blacksites}")
    else:
        logger.warning(f"[ProbeWarn] {len(blacksites)} sites need confirmation: {blacksites}")

    data = {
        AVAILABLES: ",".join(items),
        CANDIDATES: json.dumps(potentials),
    }

    persist = blacklist.get("persist", "")
    if auto_addblack and len(blacksites) > 0 and persist:
        data[persist] = merge_blacklist(regex, list(blacksites))

    return data


def parse_domains(urls: list[str]) -> dict:
    if not urls or not isinstance(urls, list):
        return {}

    domains = dict()
    for url in urls:
        domain = utils.extract_domain(url=url, include_protocal=False)
        if domain:
            domains[domain] = url

    return domains


def remove_url_param(url: str, key: str) -> str:
    url, key = utils.trim(url), utils.trim(key)
    if not url or not key:
        return url

    try:
        result = parse.urlparse(url)
        params = parse.parse_qs(result.query)
        params.pop(key, None)

        query = parse.urlencode(params, doseq=True)
        return parse.urlunparse(
            (
                result.scheme,
                result.netloc,
                result.path,
                result.params,
                query,
                result.fragment,
            )
        )
    except:
        return url


if __name__ == "__main__":
    # load .env if file exist
    utils.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--async",
        dest="run_async",
        action="store_true",
        default=False,
        help="run with asynchronous mode",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=os.environ.get("COLLECT_CONF", "").strip(),
        help="remote config file",
    )

    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=False,
        default="",
        help="final available API save file name",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        default="gpt-3.5-turbo",
        help="model name to chat with",
    )

    parser.add_argument(
        "-n",
        "--num",
        type=int,
        required=False,
        default=512,
        help="number of concurrent threads, default twice the number of CPU",
    )

    parser.add_argument(
        "-s",
        "--show",
        dest="show",
        action="store_true",
        default=False,
        help="show check progress bar",
    )

    process(args=parser.parse_args())

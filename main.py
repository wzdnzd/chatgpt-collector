import argparse
import importlib
import itertools
import json
import multiprocessing
import os
import re
import time
import traceback
from enum import Enum
from http.client import HTTPResponse
from multiprocessing.managers import DictProxy, ListProxy
from multiprocessing.synchronize import Semaphore
from typing import Any
from urllib import parse

import push
import utils
from logger import logger
from urlvalidator import isurl


class GPTProvider(Enum):
    OPENAI = 1
    AZURE = 2
    PROXIEDOPENAI = 3
    AILINK = 4
    UNKNOWN = 5


"""
/api/chat-process: https://github.com/Chanzhaoyu/chatgpt-web/blob/main/src/api/index.ts
/api/chat-stream: https://github.com/Yidadaa/ChatGPT-Next-Web/blob/main/app/api/chat-stream/route.ts
/api/chat: https://github.com/mckaywrigley/chatbot-ui/blob/main/pages/api/chat.ts
/api: https://github.com/ourongxing/chatgpt-vercel/blob/main/src/routes/api/index.ts
"""
COMMON_PATH_MODE = {
    "/api/chat-stream": [GPTProvider.OPENAI, GPTProvider.AZURE],
    "/api/chat-process": [GPTProvider.AZURE, GPTProvider.OPENAI],
    "/api/chat": [GPTProvider.PROXIEDOPENAI, GPTProvider.OPENAI, GPTProvider.AZURE],
    "/api": [GPTProvider.OPENAI, GPTProvider.AZURE],
    "/api/generateStream": [GPTProvider.OPENAI, GPTProvider.AZURE],
    "/v1/chat/completions": [GPTProvider.OPENAI],
    "/v1/chat/gpt/": [GPTProvider.AILINK],
}

DEFAULT_PROMPT = "Please tell me what is ChatGPT in English with at most 20 words"


COMMON_PAYLOAD = {
    GPTProvider.OPENAI: {
        "messages": [{"role": "user", "content": DEFAULT_PROMPT}],
        "stream": True,
        "model": "gpt-3.5-turbo",
        "temperature": 1,
        "presence_penalty": 0,
    },
    GPTProvider.AZURE: {
        "prompt": DEFAULT_PROMPT,
        "options": {},
    },
    GPTProvider.PROXIEDOPENAI: {
        "model": {
            "id": "gpt-3.5-turbo",
            "name": "GPT-3.5",
            "maxLength": 12000,
            "tokenLimit": 4000,
        },
        "messages": [{"role": "user", "content": DEFAULT_PROMPT}],
    },
    GPTProvider.AILINK: {
        "list": [
            {"role": "user", "content": DEFAULT_PROMPT},
            {"role": "assistant", "content": "..."},
        ],
        "temperature": 1,
    },
}


HEADERS = {
    "Content-Type": "application/json",
    "path": "v1/chat/completions",
    "User-Agent": utils.USER_AGENT,
}


def query_provider(name: str) -> GPTProvider:
    if not name:
        return GPTProvider.UNKNOWN
    for item in GPTProvider:
        if name.upper() == item.name:
            return item

    return GPTProvider.UNKNOWN


def execute_script(script: str, params: dict = {}) -> list[str]:
    try:
        # format: a.b.c#function or a-b.c#_function or a#function and so on
        regex = r"^([a-zA-Z0-9_]+|([0-9a-zA-Z_]+([a-zA-Z0-9_\-]+)?\.)+)[a-zA-Z0-9_\-]+#[a-zA-Z_]+[0-9a-zA-Z_]+$"
        if not re.match(regex, script):
            logger.info(
                f"[ScriptError] script execute error because script: {script} is invalidate"
            )
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
            logger.error(
                f"[ScriptError] return value error, need a list, but got a {type(sites)}"
            )
            return []

        endtime = time.time()
        logger.info(
            "[ScriptInfo] finished execute script: scripts.{}, cost: {:.3}s".format(
                script, endtime - starttime
            )
        )

        return sites
    except:
        logger.error(
            f"[ScriptError] occur error run script: {script}, message: \n{traceback.format_exc()}"
        )
        return []


def call(
    script: str, params: dict, availables: ListProxy, semaphore: Semaphore
) -> None:
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
                p = multiprocessing.Process(
                    target=call, args=(k, v, availables, semaphore)
                )
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
        cpu_count = multiprocessing.cpu_count()
        num = len(params) if len(params) <= cpu_count else cpu_count

        pool = multiprocessing.Pool(num)
        results = pool.starmap(crawl_single_page, params)
        pool.close()

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
        logger.error(
            f"[PageError] invalid task configuration, must specify url and include or regex"
        )
        return []

    try:
        content, collections = utils.http_get(url=url), set()
        if content == "":
            logger.error(f"[PageError] invalid response, site: {url}")
            return []

        regex = (
            "https?://(?:[a-zA-Z0-9\u4e00-\u9fa5\-]+\.)+[a-zA-Z0-9\u4e00-\u9fa5\-]+"
            if utils.isblank(regex)
            else regex
        )
        groups = re.findall(regex, content, flags=re.I)

        for item in groups:
            try:
                if not re.search(include, item, flags=re.I) or (
                    exclude and re.search(exclude, item, flags=re.I)
                ):
                    continue

                item = utils.url_complete(item)
                if item:
                    collections.add(item)
            except:
                logger.error(
                    f"[PageError] maybe pattern 'include' or 'exclude' exists some problems, include: {include}\texclude: {exclude}"
                )

        sites = list(collections)
        logger.info(
            f"[PageInfo] crawl page {url} finished, found {len(sites)} sites: {sites}"
        )

        uripath, mode = utils.trim(uripath).lower(), utils.trim(mode).lower()

        # regular url path
        if not re.match(
            "^(\/?\w+)+((\.)?\w+|\/)(\?(\w+=[\w\d]+(&\w+=[\w\d]+)*)+){0,1}$", uripath
        ):
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


def process(url: str) -> None:
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
                logger.warning("[ConfigWarn] no valid crawler task found")
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

            data = batch_probe(
                urls=urls,
                candidates=candidates,
                threshold=tasks.get("threshold"),
                tolerance=tasks.get("tolerance"),
                blacklist=blacklist,
            )

            if not data:
                logger.warning("[ProcessWarn] cannot found any domains")
                return

            params = [
                [v, tasks.get("persists").get(k), k, 5] for k, v in data.items() if v
            ]
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

    if "candidates" not in groups or "availables" not in groups:
        logger.error(
            "[ConfigError] persists must include 'availables' and 'candidates'"
        )
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

    if not script_tasks and not page_tasks:
        logger.error(
            "[ConfigError] cannot found any legal collect task from scripts and pages"
        )
        return {}

    threshold = max(int(data.get("threshold", 72)), 1)
    tolerance = max(int(data.get("tolerance", 3)), 1)
    data.update(
        {
            "threshold": threshold,
            "tolerance": tolerance,
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
        content = utils.http_get(url=pushtool.raw_url(v))
        if not content:
            continue
        if k == "availables":
            for site in content.split(","):
                if not utils.isblank(site):
                    candidates[site] = {"defeat": 0}

            continue
        elif k == "candidates":
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
    if address not in persists or address in ["availables", "candidates"]:
        return {}

    url = pushtool.raw_url(persists.get(address))
    content, regex = utils.http_get(url=url), ""
    if content:
        sites = list(
            set(
                [
                    x
                    for x in content.split("\n")
                    if not utils.isblank(x) and not x.startswith("#")
                ]
            )
        )
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
        if pattern and pattern.search(site):
            continue

        hostname = parse.urlparse(url=site).netloc
        url = dataset.get(hostname, "")
        if not url or len(url) < len(site):
            dataset[hostname] = site

    return list(dataset.values())


def batch_probe(
    urls: list[str],
    candidates: dict,
    threshold: int,
    tolerance: int,
    blacklist: dict = {},
) -> dict:
    if not urls and not candidates:
        return {}

    urls = urls if urls is not None else []
    candidates = candidates if candidates is not None else {}
    urls.extend(candidates.keys())

    # automatically add sites whose failure times exceed the threshold to the blacklist
    auto_addblack, regex = blacklist.get("auto", False), blacklist.get("regex", "")
    sites = intercept(sites=urls, blacklist=regex)
    logger.info(f"[ProbeSites] starting sites probe, count: {len(sites)}")

    with multiprocessing.Manager() as manager:
        collections, blacksites, potentials, processes = (
            manager.list(),
            manager.list(),
            manager.dict(),
            [],
        )
        semaphore, starttime = multiprocessing.Semaphore(max(50, 1)), time.time()
        for site in sites:
            semaphore.acquire()
            p = multiprocessing.Process(
                target=check,
                args=(
                    site,
                    candidates,
                    threshold,
                    auto_addblack,
                    collections,
                    blacksites,
                    potentials,
                    semaphore,
                    tolerance,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        availables = list(collections)
        cost = "{:.2f}s".format(time.time() - starttime)
        logger.info(
            f"[ProbeInfo] collect finished, cost: {cost}, found {len(availables)} sites"
        )

        if auto_addblack:
            logger.warning(
                f"[ProbeWarn] add {len(blacksites)} sites to blacklist: {list(blacksites)}"
            )
        else:
            logger.warning(
                f"[ProbeWarn] {len(blacksites)} sites need confirmation: {list(blacksites)}"
            )

        data = {
            "availables": ",".join(availables),
            "candidates": json.dumps(dict(potentials)),
        }

        persist = blacklist.get("persist", "")
        if auto_addblack and len(blacksites) > 0 and persist:
            data[persist] = merge_blacklist(regex, list(blacksites))

        return data


def check(
    url: str,
    candidates: dict,
    threshold: int,
    auto_addblack: bool,
    availables: ListProxy,
    blacksites: ListProxy,
    potentials: DictProxy,
    semaphore: Semaphore,
    tolerance: int,
) -> None:
    try:
        # record start time
        starttime = time.time()

        # detect connectivity
        status, apipath = judge(url=url, retry=2, tolerance=tolerance)

        # record spend time
        cost = time.time() - starttime
        domain = utils.extract_domain(url=url, include_protocal=True)
        message = (
            "[CheckInfo] finished check, site: {} status: {} cost: {:.2f}s".format(
                domain, status, cost
            )
        )
        if status:
            logger.info(message)
        else:
            logger.error(message)

        candidates = {} if candidates is None else candidates
        # reachable
        if status:
            availables.append(apipath)
            return

        # response status code is 200 but return html content
        reuslt = parse.urlparse(url)
        site, path = reuslt.netloc, reuslt.path
        if apipath:
            site = site if auto_addblack else apipath
            blacksites.append(site)
            return

        # comes from a crawler and fails any api test
        if not path or path == "/":
            return

        defeat = candidates.get(url).get("defeat", 0) + 1 if url in candidates else 1
        if defeat <= threshold:
            potentials[url] = {"defeat": defeat}
        elif auto_addblack:
            blacksites.append(parse.urlparse(url).netloc)
    finally:
        if semaphore is not None:
            semaphore.release()


def parse_url(url: str) -> tuple[bool, str, list[GPTProvider]]:
    if not isurl(url):
        return False, "", [GPTProvider.UNKNOWN]

    result = parse.urlparse(url=url)
    full, modes = result.path and result.path != "/", [GPTProvider.UNKNOWN]
    apipath = f"{result.scheme}://{result.netloc}"
    if full:
        apipath = f"{apipath}{result.path}"
        modes = COMMON_PATH_MODE.get(result.path, modes)

    if result.query:
        params = {k: v[0] for k, v in parse.parse_qs(result.query).items()}
        provider = query_provider(params.get("mode", ""))
        modes = [provider] if provider != GPTProvider.UNKNOWN else modes

    return full, apipath, modes


def generate_tasks(url: str) -> list[tuple[str, GPTProvider]]:
    full, apipath, modes = parse_url(url=url)
    if not apipath:
        return []

    flag = len(modes) > 0 and GPTProvider.UNKNOWN not in modes
    if full:
        if flag:
            return [(apipath, x) for x in modes]
        else:
            return [(apipath, x) for x in COMMON_PAYLOAD.keys()]

    # try common combinations first for speedup
    combinations, mostlikely = [], []
    for k, v in COMMON_PATH_MODE.items():
        link = f"{apipath}{k}"
        if GPTProvider.OPENAI not in v:
            v.append(GPTProvider.OPENAI)

        mostlikely.append((link, v[0]))
        combinations.extend([(link, x) for x in v[1:]])

    # combine
    combinations.extend(
        [
            (f"{apipath}{x}", y)
            for x in COMMON_PATH_MODE.keys()
            for y in COMMON_PAYLOAD.keys()
        ]
    )

    mostlikely.extend(combinations)
    # remove duplicates and return in original order
    tasks = list(set(mostlikely))
    tasks.sort(key=mostlikely.index)
    return tasks


def read_response(response: HTTPResponse, key: str = "", expected: int = 200) -> Any:
    if not response or type(response) != HTTPResponse:
        return None

    success = expected == response.getcode()
    if not success:
        return None
    try:
        content = response.read().decode("UTF8")
        data = json.loads(content)
        return data if not key else data.get(key, None)
    except:
        return None


def judge(url: str, retry: int = 2, tolerance: int = 3) -> tuple[bool, str]:
    """
    Judge whether the website is valid and sniff api path
    """
    tasks, tolerance = generate_tasks(url=url), max(tolerance, 1)
    if not tasks:
        return False, ""

    error, notfound = 0, 0
    for apipath, mode in tasks:
        body, headers = COMMON_PAYLOAD.get(mode), {"Referer": apipath}
        headers.update(HEADERS)

        if mode == GPTProvider.PROXIEDOPENAI and apipath.endswith("/api/chat"):
            prefix = apipath.rsplit("/", maxsplit=1)[0]
            response = utils.http_post_noerror(
                url=f"{prefix}/user", params={"authcode": ""}
            )
            authcode = read_response(response=response, key="authCode")

            # x-auth-code
            if authcode and type(authcode) == str:
                headers["x-auth-code"] = authcode

            response = utils.http_post_noerror(
                url=f"{prefix}/models", params={"key": ""}
            )
            models = read_response(response=response, key="")
            if models and type(models) == list:
                models.sort(key=lambda x: x.get("id", ""))
                body["model"] = models[0]

        link = f"{apipath}?mode={mode.name.lower()}"

        try:
            timeout = max(12 - 2.5 * (error + notfound), 6)
            response, exitcode = utils.http_post(
                url=apipath,
                headers=headers,
                params=body,
                retry=retry,
                timeout=timeout,
            )

            # reduce unnecessary attempts to speed up
            if exitcode > 0:
                if exitcode == 2:
                    notfound += 1
                elif exitcode == 3:
                    error += 1
            if (notfound + error) >= len(COMMON_PATH_MODE):
                logger.error(f"[JudgeError] not found any valid url path in site={url}")
                return False, ""
            if error > tolerance:
                logger.error(
                    f"[JudgeError] site={url} reached maximum allowable errors: {tolerance}"
                )
                return False, ""

            if not response or response.getcode() != 200:
                continue

            link = f"{link}&auth=true" if "x-auth-code" in headers else link
            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type:
                logger.warning(
                    f"[JudgeWarn] site=[{link}] access success but return html content"
                )
                return False, link

            allow_origin = response.headers.get("Access-Control-Allow-Origin", "*")
            if allow_origin and allow_origin != "*":
                logger.warning(
                    f"[JudeWarn] site=[{link}] access success but only support origin {allow_origin}"
                )
                return False, link

            content = response.read().decode("UTF8")
            if not content:
                continue

            # return directly without further analysis
            keywords = "ChatGPT"
            if "text/event-stream" in content_type:
                keywords += "|finish_reason|chatcmpl-"

            match = re.search(keywords, content, flags=re.I)
            return match is not None, link

            # if not re.search("ChatGPT", content, flags=re.I):
            #     logger.warning(
            #         f"[JudgeWarn] site=[{link}] access success but returned content irrelevant to the question"
            #     )
            #     return False, link

            # if not content_type or "text/plain" in content_type:
            #     if "invalid_request_error" in content:
            #         continue

            #     return True, link
            # try:
            #     index = content.rfind("\n", 0, len(content) - 2)
            #     text = content if index < 0 else content[index + 1 :]
            #     data, keys = json.loads(text), set(["id", "role", "text", "delta"])

            #     # check whether data's keys include 'id', 'role', 'text' or 'delta'
            #     if not keys.isdisjoint(set(data.keys())):
            #         return True, link
            # except:
            #     if (
            #         re.search(
            #             r'"role":(\s+)?".*",(\s+)?"id":(\s+)?"[A-Za-z0-9\-]+"|"delta":(\s+)?{"content":(\s+)?"([\s\S]*?)"',
            #             content,
            #         )
            #         or '"text":"ChatGPT is' in content
            #     ):
            #         return True, link
            #     else:
            #         logger.error(
            #             f"[JudgeError] url: {apipath} mode: {mode.name} message: {content}"
            #         )
        except Exception as e:
            logger.error(
                f"[JudgeError] url: {apipath} mode: {mode.name} message: {str(e)}"
            )

    return False, ""


if __name__ == "__main__":
    # load .env if file exist
    utils.load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=os.environ.get("SUBSCRIBE_CONF", "").strip(),
        help="remote config file",
    )

    args = parser.parse_args()
    process(url=args.config)

# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-12-09

import argparse
import gzip
import itertools
import json
import logging
import math
import os
import re
import socket
import ssl
import time
import traceback
import typing
import urllib
import urllib.error
import urllib.parse
import urllib.request
from concurrent import futures
from threading import Lock

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}

DEFAULT_QUESTION = f"Tell me what ChatGPT is in English, your answer should contain a maximum of 20 words and must start with 'ChatGPT is'!"

FILE_LOCK = Lock()


PATH = os.path.abspath(os.path.dirname(__file__))


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(os.path.join(PATH, "search.log")), logging.StreamHandler()],
)


def search_github_web(query: str, session: str, page: int) -> str:
    """use github web search instead of rest api due to it not support regex syntax"""

    if page <= 0 or isblank(session) or isblank(query):
        return ""

    url = f"https://github.com/search?o=desc&p={page}&type=code&q={query}"
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Referer": "https://github.com",
        "User-Agent": USER_AGENT,
        "Cookie": f"user_session={session}",
    }

    content = http_get(url=url, headers=headers)
    if re.search(r"<h1>Sign in to GitHub</h1>", content, flags=re.I):
        logging.error("[GithubCrawl] session has expired, please provide a valid session and try again")
        return ""

    return content


def search_github_api(query: str, token: str, page: int = 1, peer_page: int = 100) -> list[str]:
    if isblank(token) or isblank(query):
        return []

    peer_page, page = min(max(peer_page, 1), 100), max(1, page)
    url = f"https://api.github.com/search/code?q={query}&sort=indexed&order=desc&per_page={peer_page}&page={page}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    content = http_get(url=url, headers=headers, interval=1)
    if isblank(content):
        return []
    try:
        items = json.loads(content).get("items", [])
        links = set()

        for item in items:
            if not item or type(item) != dict:
                continue

            link = item.get("html_url", "")
            if isblank(link):
                continue
            links.add(link)

        return list(links)
    except:
        return []


def get_total_num(query: str, token: str) -> int:
    if isblank(token) or isblank(query):
        return 0

    url = f"https://api.github.com/search/code?q={query}&sort=indexed&order=desc&per_page=20&page=1"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    content = http_get(url=url, headers=headers, interval=1)
    try:
        data = json.loads(content)
        return data.get("total_count", 0)
    except:
        logging.error(f"[GithubCrawl] failed to get total number of items with query: {query}")
        return 0


def search_code(query: str, session: str, page: int, with_api: bool, peer_page: int) -> list[str]:
    keyword = trim(query)
    if not keyword:
        return []

    if with_api:
        return search_github_api(query=keyword, token=session, page=page, peer_page=peer_page)

    content = search_github_web(query=keyword, session=session, page=page)
    if isblank(content):
        return []

    try:
        regex = r'href="(/\S+/blob/.*?)#L\d+"'
        groups = re.findall(regex, content, flags=re.I)
        uris = list(set(groups)) if groups else []
        links = set()

        for uri in uris:
            links.add(f"https://github.com{uri}")

        return list(links)
    except:
        return []


def extract(url: str, regex: str, retries: int = 3) -> list[str]:
    if not isinstance(url, str) or not isinstance(regex, str):
        return []

    content = http_get(url=url, retries=retries, interval=1)
    try:
        groups = re.findall(regex, content, flags=re.I)
        items = set()
        for x in groups:
            key = trim(x)
            if key:
                items.add(key)

        return list(items)
    except:
        logging.error(traceback.format_exc())
        return []


def scan(
    regex: str,
    session: str,
    check: typing.Callable,
    filename: str,
    query: str = "",
    with_api: bool = False,
    page_num: int = -1,
    num_thread: int = None,
) -> None:
    regex, session = trim(regex), trim(session)
    if not regex or not session:
        logging.error(f"[Scan] skip to scan due to regex or session is empty")
        return

    if not isinstance(check, typing.Callable):
        logging.error(f"[Scan] check must be a callable function")
        return

    filename = trim(filename)
    if not filename:
        logging.error(f"[Scan] filename cannot be empty")
        return

    query = trim(query) or f"/{regex}/"
    keyword = urllib.parse.quote(query, safe="")
    peer_page, count = 100, 5

    if with_api:
        total = get_total_num(query=keyword, token=session)
        logging.info(f"[Scan] found {total} items with query: {query}")

        if total > 0:
            count = math.ceil(total / peer_page)

    page_num = count if page_num < 0 or page_num > count else page_num
    if page_num <= 0:
        logging.error(f"[Scan] page number must be greater than 0")
        return

    queries = [[keyword, session, x, with_api, peer_page] for x in range(1, page_num + 1)]
    candidates = multi_thread_run(
        func=search_code,
        tasks=queries,
        num_threads=num_thread,
    )
    links = list(set(itertools.chain.from_iterable(candidates)))
    if not links:
        logging.warning(f"[Scan] cannot found any link with query: {query}")
        return

    logging.info(f"[Scan] start to extract candidates from {len(links)} links")

    tasks = [[x, regex, 3] for x in links if x]
    result = multi_thread_run(func=extract, tasks=tasks, num_threads=num_thread)

    records = list()
    filepath = os.path.join(PATH, filename)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        # load exists keys
        with open(filepath, "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                text = trim(line)
                if not text or text.startswith(";") or text.startswith("#"):
                    continue

                records.append(text)

        logging.info(f"[Scan] loaded {len(records)} exists keys from file {filepath}")

        # backup up exists file with current time
        words = filename.rsplit(".", maxsplit=1)
        filename = f"{words[0]}-{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        if len(words) > 1:
            filename += f".{words[1]}"

        os.rename(filepath, os.path.join(PATH, filename))

    records.extend(list(itertools.chain.from_iterable(result)))
    if not records:
        logging.warning(f"[Scan] cannot extract any candidate with query: {query}")
        return

    items = list(set(records))
    logging.info(f"[Scan] start to verify {len(items)} potential keys")

    masks = multi_thread_run(func=check, tasks=items, num_threads=num_thread)
    keys = [items[i] for i in range(len(masks)) if masks[i]]

    if not keys:
        logging.warning(f"[Scan] finished, cannot found any key with query: {query}")
    else:
        saved = write_file(directory=filepath, lines=keys)
        if not saved:
            logging.error(f"[Scan] failed to save keys to file: {filepath}, keys: {keys}")

        logging.info(f"[Scan] finished, found {len(keys)} valid keys, save them to file: {filepath}")


def chat(url: str, headers: dict, model: str = "", params: dict = None, retries: int = 2, timeout: int = 10) -> dict:
    url, model = trim(url), trim(model)
    if not url:
        logging.error(f"[Check] url cannot be empty")
        return None

    if not isinstance(headers, dict):
        logging.error(f"[Check] headers must be a dict")
        return None
    elif len(headers) == 0:
        headers["content-type"] = "application/json"

    if not params or not isinstance(params, dict):
        if not model:
            logging.error(f"[Check] model cannot be empty")
            return None

        params = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": DEFAULT_QUESTION}],
        }

    payload = json.dumps(params).encode("utf8")
    timeout = max(1, timeout)
    retries = max(1, retries)
    data, attempt = None, 0

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    while attempt < retries:
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as response:
                content = response.read()
                data = json.loads(content)
                break
        except Exception:
            pass

        attempt += 1
        time.sleep(1)

    return data


def scan_openai_keys(session: str, with_api: bool = False, page_num: int = -1) -> None:
    def verify(token: str) -> bool:
        token = trim(token)
        if not token:
            return False

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"content-type": "application/json", "authorization": f"Bearer {token}"}
        data = chat(url=url, headers=headers, model="gpt-4o-mini")
        return data is not None

    # TODO: optimize query syntax for github api
    query = '"OPENAI_API_KEY=sk-"' if with_api else ""
    regex = r"sk-[a-zA-Z0-9]{48}"
    filename = "openai-keys.txt"

    scan(
        regex=regex,
        session=session,
        check=verify,
        filename=filename,
        query=query,
        with_api=with_api,
        page_num=page_num,
    )


def scan_anthropic_keys(session: str, with_api: bool = False, page_num: int = -1) -> None:
    def verify(token: str) -> bool:
        token = trim(token)
        if not token:
            return False

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "content-type": "application/json",
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
        }
        data = chat(url=url, headers=headers, model="claude-3-5-sonnet-20241022")
        return data is not None

    query = '"sk-ant-api03-"' if with_api else ""
    regex = r"sk-ant-api03-[a-zA-Z0-9_\-]{86}-[a-zA-Z0-9_]{8}"
    filename = "anthropic-keys.txt"

    scan(
        regex=regex,
        session=session,
        check=verify,
        filename=filename,
        query=query,
        with_api=with_api,
        page_num=page_num,
    )


def scan_gemini_keys(session: str, with_api: bool = False, page_num: int = -1) -> None:
    def verify(token: str) -> bool:
        token = trim(token)
        if not token:
            return False

        model = "gemini-exp-1206"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={token}"
        headers = {"content-type": "application/json"}
        params = {"contents": [{"role": "user", "parts": [{"text": DEFAULT_QUESTION}]}]}

        data = chat(url=url, headers=headers, params=params)
        return data is not None

    query = '"GEMINI_API_KEY=AIzaSy"' if with_api else "/GEMINI_API_KEY=[a-zA-Z0-9_]{39}/"
    regex = r"GEMINI_API_KEY=([a-zA-Z0-9_]{39})"
    filename = "gemini-keys.txt"

    scan(
        regex=regex,
        session=session,
        check=verify,
        filename=filename,
        query=query,
        with_api=with_api,
        page_num=page_num,
    )


def trim(text: str) -> str:
    if not text or type(text) != str:
        return ""

    return text.strip()


def isblank(text: str) -> bool:
    return not text or type(text) != str or not text.strip()


def http_get(
    url: str,
    headers: dict = None,
    params: dict = None,
    retries: int = 3,
    interval: float = 0,
    timeout: float = 10,
) -> str:
    if isblank(text=url):
        logging.error(f"invalid url: {url}")
        return ""

    if retries <= 0:
        return ""

    headers = DEFAULT_HEADERS if not headers else headers

    interval = max(0, interval)
    timeout = max(1, timeout)
    try:
        url = encoding_url(url=url)
        if params and isinstance(params, dict):
            data = urllib.parse.urlencode(params)
            if "?" in url:
                url += f"&{data}"
            else:
                url += f"?{data}"

        request = urllib.request.Request(url=url, headers=headers)
        response = urllib.request.urlopen(request, timeout=timeout, context=CTX)
        content = response.read()
        status_code = response.getcode()
        try:
            content = str(content, encoding="utf8")
        except:
            content = gzip.decompress(content).decode("utf8")
        if status_code != 200:
            return ""

        return content
    except urllib.error.URLError as e:
        if isinstance(e.reason, (socket.timeout, ssl.SSLError)):
            time.sleep(interval)
            return http_get(
                url=url,
                headers=headers,
                params=params,
                retries=retries - 1,
                interval=interval,
                timeout=timeout,
            )
        else:
            return ""
    except Exception as e:
        time.sleep(interval)
        return http_get(
            url=url,
            headers=headers,
            params=params,
            retries=retries - 1,
            interval=interval,
            timeout=timeout,
        )


def multi_thread_run(func: typing.Callable, tasks: list, num_threads: int = None) -> list:
    if not func or not tasks or not isinstance(tasks, list):
        return []

    if num_threads is None or num_threads <= 0:
        num_threads = min(len(tasks), (os.cpu_count() or 1) * 2)

    funcname = getattr(func, "__name__", repr(func))

    results, starttime = [None] * len(tasks), time.time()
    with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        if isinstance(tasks[0], (list, tuple)):
            collections = {executor.submit(func, *param): i for i, param in enumerate(tasks)}
        else:
            collections = {executor.submit(func, param): i for i, param in enumerate(tasks)}

        items = futures.as_completed(collections)
        for future in items:
            try:
                result = future.result()
                index = collections[future]
                results[index] = result
            except:
                logging.error(
                    f"function {funcname} execution generated an exception, message:\n{traceback.format_exc()}"
                )

    logging.info(
        f"[Concurrent] multi-threaded execute [{funcname}] finished, count: {len(tasks)}, cost: {time.time()-starttime:.2f}s"
    )

    return results


def encoding_url(url: str) -> str:
    if not url:
        return ""

    url = url.strip()
    cn_chars = re.findall("[\u4e00-\u9fa5]+", url)
    if not cn_chars:
        return url

    punycodes = list(map(lambda x: "xn--" + x.encode("punycode").decode("utf-8"), cn_chars))
    for c, pc in zip(cn_chars, punycodes):
        url = url[: url.find(c)] + pc + url[url.find(c) + len(c) :]

    return url


def write_file(directory: str, lines: str | list) -> bool:
    if not directory or not lines or not isinstance(lines, (str, list)):
        logging.error(f"filename or lines is invalid, filename: {directory}")
        return False

    try:
        if not isinstance(lines, str):
            lines = "\n".join(lines)

        filepath = os.path.abspath(os.path.dirname(directory))
        os.makedirs(filepath, exist_ok=True)

        # waitting for lock
        FILE_LOCK.acquire(30)

        with open(directory, "a+", encoding="UTF8") as f:
            f.write(lines + "\n")
            f.flush()

        # release lock
        FILE_LOCK.release()

        return True
    except:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--all",
        dest="all",
        action="store_true",
        default=False,
        help="execute all supported scan tasks",
    )

    parser.add_argument(
        "-c",
        "--claude",
        dest="claude",
        action="store_true",
        default=False,
        help="scan claude api keys",
    )

    parser.add_argument(
        "-g",
        "--gemini",
        dest="gemini",
        action="store_true",
        default=False,
        help="scan gemini api keys",
    )

    parser.add_argument(
        "-n",
        "--num",
        type=int,
        required=False,
        default=-1,
        help="number of pages to scan. default is -1, mean scan all pages",
    )

    parser.add_argument(
        "-o",
        "--openai",
        dest="openai",
        action="store_true",
        default=False,
        help="scan openai api keys",
    )

    parser.add_argument(
        "-r",
        "--rest",
        dest="rest",
        action="store_true",
        default=False,
        help="search code through github rest api",
    )

    parser.add_argument(
        "-s",
        "--session",
        type=str,
        required=True,
        default="",
        help="github user session, key name is 'user_session'",
    )

    args = parser.parse_args()
    session = trim(args.session)

    if not session:
        logging.error(f"session cannot be empty")
        exit(1)

    if args.all or args.claude:
        logging.info(f"start to scan claude api keys")
        scan_anthropic_keys(session=session, with_api=args.rest, page_num=args.num)

    if args.all or args.gemini:
        logging.info(f"start to scan gemini api keys")
        scan_gemini_keys(session=session, with_api=args.rest, page_num=args.num)

    if args.all or args.openai:
        logging.info(f"start to scan openai api keys")
        scan_openai_keys(session=session, with_api=args.rest, page_num=args.num)

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
import random
import re
import ssl
import time
import traceback
import typing
import urllib
import urllib.error
import urllib.parse
import urllib.request
from concurrent import futures
from dataclasses import dataclass, field
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

DEFAULT_QUESTION = "Hello"


# error http status code that do not need to retry
NO_RETRY_ERROR_CODES = {400, 401, 404, 422}


FILE_LOCK = Lock()


PATH = os.path.abspath(os.path.dirname(__file__))


logging.basicConfig(
    format="%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(os.path.join(PATH, "search.log")), logging.StreamHandler()],
)


@dataclass
class KeyDetail(object):
    # token
    key: str

    # models that the key can access
    models: list[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        if not isinstance(other, KeyDetail):
            return False

        return self.key == other.key


@dataclass
class Condition(object):
    # pattern for extract key from code
    regex: str

    # search keyword or pattern
    query: str = ""

    def __hash__(self):
        return hash((self.query, self.regex))

    def __eq__(self, other):
        if not isinstance(other, Condition):
            return False

        return self.query == other.query and self.regex == other.regex


class Provider(object):
    def __init__(
        self,
        name: str,
        base_url: str,
        completion_path: str,
        model_path: str,
        default_model: str,
        conditions: Condition | list[Condition],
        **kwargs,
    ):
        name = str(name)
        if not name:
            raise ValueError("provider name cannot be empty")

        default_model = trim(default_model)
        if not default_model:
            raise ValueError("default_model cannot be empty")

        base_url = trim(base_url)
        if not re.match(r"^https?:\/\/([\w\-_]+\.[\w\-_]+)+", base_url):
            raise ValueError("base_url must be a valid url")

        # see: https://stackoverflow.com/questions/10893374/python-confusions-with-urljoin
        if not base_url.endswith("/"):
            base_url += "/"

        filename = name.replace(" ", "-").lower()

        # provider name
        self.name = name

        # filename for keys
        self.keys_filename = f"{filename}-keys.txt"

        # filename for extract keys
        self.material_filename = f"{filename}-material.txt"

        # filename for summary
        self.summary_filename = f"{filename}-summary.json"

        # filename for links included keys
        self.links_filename = f"{filename}-links.txt"

        # base url for llm service api
        self.base_url = base_url

        # path for completion api
        self.completion_path = trim(completion_path).removeprefix("/")

        # path for model list api
        self.model_path = trim(model_path).removeprefix("/")

        # default model for completion api used to verify token
        self.default_model = default_model

        conditions = (
            [conditions]
            if isinstance(conditions, Condition)
            else ([] if not isinstance(conditions, list) else conditions)
        )

        items = set()
        for condition in conditions:
            if not isinstance(condition, Condition) or not condition.regex:
                logging.warning(f"invalid condition: {condition}, skip it")
                continue

            items.add(condition)

        # search and extract keys conditions
        self.conditions = list(items)

        # additional parameters for provider
        self.extras = kwargs

    def _get_headers(self, token: str) -> dict:
        raise NotImplementedError

    def check(self, token):
        url = urllib.parse.urljoin(self.base_url, self.completion_path)
        headers = self._get_headers(token=token)
        if not headers:
            return False

        data = chat(url=url, headers=headers, model=self.default_model)
        return data is not None

    def list_models(self, token) -> list[str]:
        raise NotImplementedError


class OpenAILikeProvider(Provider):
    def __init__(
        self,
        name: str,
        base_url: str,
        default_model: str,
        conditions: list[Condition],
        completion_path: str = "/v1/chat/completions",
        model_path: str = "/v1/models",
        **kwargs,
    ):
        super().__init__(name, base_url, completion_path, model_path, default_model, conditions, **kwargs)

    def _get_headers(self, token):
        token = trim(token)
        if not token:
            return None

        return {"content-type": "application/json", "authorization": f"Bearer {token}"}

    def list_models(self, token):
        headers = self._get_headers(token=token)
        if not headers or not self.model_path:
            return []

        url = urllib.parse.urljoin(self.base_url, self.model_path)
        content = http_get(url=url, headers=headers, interval=1)
        if not content:
            return []

        try:
            result = json.loads(content)
            return [x.get("id", "") for x in result.get("data", [])]
        except:
            logging.error(f"failed to parse models from response: {content}")
            return []


class OpenAIProvider(OpenAILikeProvider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "gpt-4o-mini"
        base_url = "https://api.openai.com"

        super().__init__("openai", base_url, default_model, conditions)


class GroqProvider(OpenAILikeProvider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "llama3-8b-8192"
        base_url = "https://api.groq.com/openai"

        super().__init__("groq", base_url, default_model, conditions)


class AnthropicProvider(Provider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "claude-3-5-sonnet-latest"
        super().__init__("anthropic", "https://api.anthropic.com", "/v1/messages", "", default_model, conditions)

    def _get_headers(self, token):
        token = trim(token)
        if not token:
            return None

        return {"content-type": "application/json", "x-api-key": token, "anthropic-version": "2023-06-01"}

    def list_models(self, token):
        token = trim(token)
        if not token:
            return []

        # see: https://docs.anthropic.com/en/docs/about-claude/models
        return [
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]


class GeminiProvider(Provider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "gemini-exp-1206"
        base_url = "https://generativelanguage.googleapis.com"
        sub_path = "/v1beta/models"

        super().__init__("gemini", base_url, sub_path, sub_path, default_model, conditions)

    def _get_headers(self, token):
        return {"content-type": "application/json"}

    def check(self, token):
        token = trim(token)
        if not token:
            return False

        url = f"{urllib.parse.urljoin(self.base_url, self.completion_path)}/{self.default_model}:generateContent?key={token}"
        params = {"contents": [{"role": "user", "parts": [{"text": DEFAULT_QUESTION}]}]}

        data = chat(url=url, headers=self._get_headers(token=token), params=params)
        return data is not None

    def list_models(self, token):
        token = trim(token)
        if not token:
            return []

        url = urllib.parse.urljoin(self.base_url, self.model_path) + f"?key={token}"
        content = http_get(url=url, headers=self._get_headers(token=token), interval=1)
        if not content:
            return []

        try:
            data = json.loads(content)
            models = data.get("models", [])
            return [x.get("name", "").removeprefix("models/") for x in models]
        except:
            logging.error(f"failed to parse models from response: {content}")
            return []


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
    """rate limit: 10RPM"""
    if isblank(token) or isblank(query):
        return []

    peer_page, page = min(max(peer_page, 1), 100), max(1, page)
    url = f"https://api.github.com/search/code?q={query}&sort=indexed&order=desc&per_page={peer_page}&page={page}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    content = http_get(url=url, headers=headers, interval=2, timeout=30)
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


def batch_search_code(
    session: str,
    query: str,
    with_api: bool = False,
    page_num: int = -1,
    thread_num: int = None,
    fast: bool = False,
) -> list[str]:
    session, query = trim(session), trim(query)
    if not query or not session:
        logging.error(f"[Search] skip to search due to query or session is empty")
        return []

    keyword = urllib.parse.quote(query, safe="")
    peer_page, count = 100, 5

    if with_api:
        total = get_total_num(query=keyword, token=session)
        logging.info(f"[Search] found {total} items with query: {query}")

        if total > 0:
            count = math.ceil(total / peer_page)

    # see: https://stackoverflow.com/questions/37602893/github-search-limit-results
    # the search api will return up to 1000 results per query (including pagination, peer_page: 100)
    # and web search is limited to 5 pages
    # so maxmum page num is 10
    page_num = min(count if page_num < 0 or page_num > count else page_num, 10)
    if page_num <= 0:
        logging.error(f"[Search] page number must be greater than 0")
        return []

    links = list()
    if fast:
        # concurrent requests are easy to fail but faster
        queries = [[keyword, session, x, with_api, peer_page] for x in range(1, page_num + 1)]
        candidates = multi_thread_run(
            func=search_code,
            tasks=queries,
            thread_num=thread_num,
        )
        links = list(set(itertools.chain.from_iterable(candidates)))
    else:
        # sequential requests are more stable but slower
        potentials = set()
        for i in range(1, page_num + 1):
            urls = search_code(query=keyword, session=session, page=i, with_api=with_api, peer_page=peer_page)
            potentials.update([x for x in urls if x])

            # avoid github api rate limit: 10RPM
            time.sleep(random.randint(6, 12))

        links = list(potentials)
    if not links:
        logging.warning(f"[Search] cannot found any link with query: {query}")

    return links


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
    session: str,
    provider: Provider,
    with_api: bool = False,
    page_num: int = -1,
    thread_num: int = None,
    fast: bool = False,
    skip: bool = False,
) -> None:
    if not isinstance(provider, Provider):
        return

    keys_filename = trim(provider.keys_filename)
    if not keys_filename:
        logging.error(f"[Scan] {provider.name}: keys filename cannot be empty")
        return

    records = set()
    filepath = os.path.join(PATH, keys_filename)
    material = os.path.join(PATH, provider.material_filename)
    links_filename = os.path.join(PATH, provider.links_filename)

    if os.path.exists(filepath) and os.path.isfile(filepath):
        # load exists valid keys
        records.update(read_file(filepath=filepath))
        logging.info(f"[Scan] {provider.name}: loaded {len(records)} exists keys from file {filepath}")

        # backup up exists file with current time
        words = keys_filename.rsplit(".", maxsplit=1)
        keys_filename = f"{words[0]}-{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        if len(words) > 1:
            keys_filename += f".{words[1]}"

        os.rename(filepath, os.path.join(PATH, keys_filename))

    if os.path.exists(material) and os.path.isfile(material):
        # load potential keys from material file
        records.update(read_file(filepath=material))

    if not skip and provider.conditions:
        new_keys, start_time = set(), time.time()
        for condition in provider.conditions:
            if not isinstance(condition, Condition):
                continue

            query, regex = condition.query, condition.regex
            logging.info(f"[Scan] {provider.name}: start to search new keys with query: {query}, regex: {regex}")

            candidates = recall(
                regex=regex,
                session=session,
                query=query,
                with_api=with_api,
                page_num=page_num,
                thread_num=thread_num,
                fast=fast,
                links_file=links_filename,
            )

            if candidates:
                new_keys.update(candidates)

        # merge new keys with exists keys
        records.update(new_keys)

        cost = time.time() - start_time
        count, total = len(provider.conditions), len(new_keys)
        logging.info(f"[Scan] {provider.name}: cost {cost:.2f}s to search {count} conditions, got {total} new keys")

    if not records:
        logging.warning(f"[Scan] {provider.name}: cannot extract any candidate with conditions: {provider.conditions}")
        return

    items = list(records)

    # save all potential keys to material file
    if not write_file(directory=material, lines=items):
        logging.error(f"[Scan] {provider.name}: failed to save potential keys to file: {material}, keys: {items}")

    logging.info(f"[Scan] {provider.name}: start to verify {len(items)} potential keys")

    masks = multi_thread_run(func=provider.check, tasks=items, thread_num=thread_num)
    keys = [items[i] for i in range(len(masks)) if masks[i]]

    if not keys:
        logging.warning(f"[Scan] {provider.name}: cannot found any key with conditions: {provider.conditions}")
    else:
        if not write_file(directory=filepath, lines=keys):
            logging.error(f"[Scan] {provider.name}: failed to save keys to file: {filepath}, keys: {keys}")
        else:
            logging.info(f"[Scan] {provider.name}: found {len(keys)} valid keys, save them to file: {filepath}")

        # list supported models for each key
        models = multi_thread_run(func=provider.list_models, tasks=keys, thread_num=thread_num)
        data = {keys[i]: models[i] or [] for i in range(len(keys))}
        summary_path = os.path.join(PATH, provider.summary_filename)

        if write_file(directory=summary_path, lines=json.dumps(data, ensure_ascii=False, indent=4)):
            logging.info(f"[Scan] {provider.name}: saved {len(keys)} keys summary to file: {summary_path}")
        else:
            logging.error(f"[Scan] {provider.name}: failed to save keys summary to file: {summary_path}, data: {data}")


def recall(
    regex: str,
    session: str,
    query: str = "",
    with_api: bool = False,
    page_num: int = -1,
    thread_num: int = None,
    fast: bool = False,
    links_file: str = "",
) -> list[str]:
    regex = trim(regex)
    if not regex:
        logging.error(f"[Recall] skip to recall due to regex is empty")
        return []

    links = set()
    links_file = os.path.abspath(trim(links_file))
    if os.path.exists(links_file) and os.path.isfile(links_file):
        # load exists links from persisted file
        lines = read_file(filepath=links_file)
        for text in lines:
            if not re.match(r"^https?://", text, flags=re.I):
                text = f"http://{text}"

            links.add(text)

    session = trim(session)
    query = trim(query) or f"/{regex}/"
    if session:
        sources = batch_search_code(
            session=session,
            query=query,
            with_api=with_api,
            page_num=page_num,
            thread_num=thread_num,
            fast=fast,
        )
        if sources:
            links.update(sources)

    if not links:
        logging.warning(f"[Recall] cannot found any link with query: {query}")
        return []

    # save links to file
    if links_file and not write_file(directory=links_file, lines=list(links), overwrite=True):
        logging.warning(f"[Recall] failed to save links to file: {links_file}, links: {links}")

    logging.info(f"[Recall] start to extract candidates from {len(links)} links")

    tasks = [[x, regex, 3] for x in links if x]
    result = multi_thread_run(func=extract, tasks=tasks, thread_num=thread_num)

    return list(itertools.chain.from_iterable(result))


def chat(url: str, headers: dict, model: str = "", params: dict = None, retries: int = 2, timeout: int = 10) -> dict:
    url, model = trim(url), trim(model)
    if not url:
        logging.error(f"[Chat] url cannot be empty")
        return None

    if not isinstance(headers, dict):
        logging.error(f"[Chat] headers must be a dict")
        return None
    elif len(headers) == 0:
        headers["content-type"] = "application/json"

    if not params or not isinstance(params, dict):
        if not model:
            logging.error(f"[Chat] model cannot be empty")
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
        except urllib.error.HTTPError as e:
            if e.code != 401:
                logging.error(
                    f"[Chat] failed to request url: {url}, headers: {headers}, status code: {e.code}, message: {e.reason}"
                )
            if e.code in NO_RETRY_ERROR_CODES:
                break
        except Exception:
            logging.error(f"[Chat] failed to request url: {url}, headers: {headers}, message: {traceback.format_exc()}")

        attempt += 1
        time.sleep(1)

    return data


def scan_anthropic_keys(
    session: str,
    with_api: bool = False,
    page_num: int = -1,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
) -> None:
    query = '"sk-ant-api03-"' if with_api else ""
    regex = r"sk-ant-api03-[a-zA-Z0-9_\-]{93}AA"

    conditions = [Condition(query=query, regex=regex)]
    default_model = "claude-3-5-sonnet-latest"
    provider = AnthropicProvider(conditions=conditions, default_model=default_model)

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        page_num=page_num,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
    )


def scan_gemini_keys(
    session: str,
    with_api: bool = False,
    page_num: int = -1,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
) -> None:
    query = '/AIzaSy[a-zA-Z0-9_\-]{33}/ AND content:"gemini"'
    if with_api:
        query = '"AIzaSy" AND "gemini"'

    regex = r"AIzaSy[a-zA-Z0-9_\-]{33}"

    conditions = [Condition(query=query, regex=regex)]
    default_model = "gemini-1.5-pro-latest"
    provider = GeminiProvider(conditions=conditions, default_model=default_model)

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        page_num=page_num,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
    )


def scan_openai_keys(
    session: str,
    with_api: bool = False,
    page_num: int = -1,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
) -> None:
    # TODO: optimize query syntax for github api
    query = '"T3BlbkFJ"' if with_api else ""
    regex = r"sk(?:-proj)?-[a-zA-Z0-9]{20}T3BlbkFJ[a-zA-Z0-9]{20}|sk-proj-(?:[a-zA-Z0-9_\-]{91}|[a-zA-Z0-9_\-]{123}|[a-zA-Z0-9_\-]{155})A"

    conditions = [Condition(query=query, regex=regex)]
    provider = OpenAIProvider(conditions=conditions, default_model="gpt-4o-mini")

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        page_num=page_num,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
    )


def scan_groq_keys(
    session: str,
    with_api: bool = False,
    page_num: int = -1,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
) -> None:
    query = '"WGdyb3FY"' if with_api else ""
    regex = r"gsk_[a-zA-Z0-9]{20}WGdyb3FY[a-zA-Z0-9]{24}"

    conditions = [Condition(query=query, regex=regex)]
    provider = GroqProvider(conditions=conditions, default_model="llama3-8b-8192")

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        page_num=page_num,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
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
    except urllib.error.HTTPError as e:
        logging.debug(f"failed to request url: {url}, status code: {e.code}, message: {e.reason}")

        if e.code in NO_RETRY_ERROR_CODES:
            return ""
    except:
        logging.debug(f"failed to request url: {url}, message: {traceback.format_exc()}")

    time.sleep(interval)
    return http_get(
        url=url,
        headers=headers,
        params=params,
        retries=retries - 1,
        interval=interval,
        timeout=timeout,
    )


def multi_thread_run(func: typing.Callable, tasks: list, thread_num: int = None) -> list:
    if not func or not tasks or not isinstance(tasks, list):
        return []

    if thread_num is None or thread_num <= 0:
        thread_num = min(len(tasks), (os.cpu_count() or 1) * 2)

    funcname = getattr(func, "__name__", repr(func))

    results, starttime = [None] * len(tasks), time.time()
    with futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
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


def read_file(filepath: str) -> list[str]:
    filepath = trim(filepath)
    if not filepath:
        logging.error(f"filepath cannot be empty")
        return []

    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        logging.error(f"file not found: {filepath}")
        return []

    lines = list()
    with open(filepath, "r", encoding="utf8") as f:
        for line in f.readlines():
            text = trim(line)
            if not text or text.startswith(";") or text.startswith("#"):
                continue

            lines.append(text)

    return lines


def write_file(directory: str, lines: str | list, overwrite: bool = True) -> bool:
    if not directory or not lines or not isinstance(lines, (str, list)):
        logging.error(f"filename or lines is invalid, filename: {directory}")
        return False

    try:
        if not isinstance(lines, str):
            lines = "\n".join(lines)

        filepath = os.path.abspath(os.path.dirname(directory))
        os.makedirs(filepath, exist_ok=True)

        mode = "w+" if overwrite else "a+"

        # waitting for lock
        FILE_LOCK.acquire(30)

        with open(directory, mode=mode, encoding="UTF8") as f:
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
        "-e",
        "--elide",
        dest="elide",
        action="store_true",
        default=False,
        help="skip search new keys from github",
    )

    parser.add_argument(
        "-f",
        "--fast",
        dest="fast",
        action="store_true",
        default=False,
        help="concurrent request github rest api to search code for speed up but easy to fail",
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
        "-l",
        "--llama",
        dest="llama",
        action="store_true",
        default=False,
        help="scan groq api keys",
    )

    parser.add_argument(
        "-n",
        "--num",
        type=int,
        required=False,
        default=-1,
        help="concurrent thread number. default is -1, mean auto select",
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
        "-p",
        "--pages",
        type=int,
        required=False,
        default=-1,
        help="number of pages to scan. default is -1, mean scan all pages",
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
        required=False,
        default="",
        help="github token if use rest api else user session key named 'user_session'",
    )

    args = parser.parse_args()
    session = trim(args.session)

    if args.all or args.claude:
        scan_anthropic_keys(
            session=session,
            with_api=args.rest,
            page_num=args.pages,
            thread_num=args.num,
            fast=args.fast,
            skip=args.elide,
        )

    if args.all or args.gemini:
        scan_gemini_keys(
            session=session,
            with_api=args.rest,
            page_num=args.pages,
            thread_num=args.num,
            fast=args.fast,
            skip=args.elide,
        )

    if args.all or args.llama:
        scan_groq_keys(
            session=session,
            with_api=args.rest,
            page_num=args.pages,
            thread_num=args.num,
            fast=args.fast,
            skip=args.elide,
        )

    if args.all or args.openai:
        scan_openai_keys(
            session=session,
            with_api=args.rest,
            page_num=args.pages,
            thread_num=args.num,
            fast=args.fast,
            skip=args.elide,
        )

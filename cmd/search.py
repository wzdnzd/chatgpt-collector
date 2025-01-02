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
from dataclasses import dataclass, field
from enum import Enum, unique
from functools import lru_cache
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


DEFAULT_COMPLETION_PATH = "/v1/chat/completions"


DEFAULT_MODEL_PATH = "/v1/models"

# error http status code that do not need to retry
NO_RETRY_ERROR_CODES = {400, 401, 402, 404, 422}


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

    # available
    available: bool = False

    # models that the key can access
    models: list[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        if not isinstance(other, KeyDetail):
            return False

        return self.key == other.key


@dataclass
class Service(object):
    # server address
    address: str = ""

    # application name or id
    endpoint: str = ""

    # api token
    key: str = ""

    # model name
    model: str = ""

    def __hash__(self):
        return hash((self.address, self.endpoint, self.key, self.model))

    def __eq__(self, other):
        if not isinstance(other, Service):
            return False

        return (
            self.address == other.address
            and self.endpoint == other.endpoint
            and self.key == other.key
            and self.model == other.model
        )

    def serialize(self) -> str:
        if not self.address and not self.endpoint and not self.model:
            return self.key

        data = dict()
        if self.address:
            data["address"] = self.address
        if self.endpoint:
            data["endpoint"] = self.endpoint
        if self.key:
            data["key"] = self.key
        if self.model:
            data["model"] = self.model

        return "" if not data else json.dumps(data)

    @classmethod
    def deserialize(cls, text: str) -> "Service":
        if not text:
            return None

        try:
            item = json.loads(text)
            return cls(
                address=item.get("address", ""),
                endpoint=item.get("endpoint", ""),
                key=item.get("key", ""),
                model=item.get("model", ""),
            )
        except:
            return cls(key=text)


@unique
class ErrorReason(Enum):
    # no error
    NONE = 1

    # insufficient_quota
    NO_QUOTA = 2

    # rate_limit_exceeded
    RATE_LIMITED = 3

    # model_not_found
    NO_MODEL = 4

    # account_deactivated
    EXPIRED_KEY = 5

    # invalid_api_key
    INVALID_KEY = 6

    # unsupported_country_region_territory
    NO_ACCESS = 7

    # server_error
    SERVER_ERROR = 8

    # bad request
    BAD_REQUEST = 9

    # unknown error
    UNKNOWN = 10


@dataclass
class CheckResult(object):
    # whether the key can be used now
    available: bool = False

    # error message if the key cannot be used
    reason: ErrorReason = ErrorReason.UNKNOWN

    def ok():
        return CheckResult(available=True, reason=ErrorReason.NONE)

    def fail(reason: ErrorReason):
        return CheckResult(available=False, reason=reason)


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

        # see: https://stackoverflow.com/questions/10893374/python-confusions-with-urljoin
        if base_url and not base_url.endswith("/"):
            base_url += "/"

        # provider name
        self.name = name

        # directory
        self.directory = re.sub(r"[^a-zA-Z0-9_\-]", "-", name, flags=re.I).lower()

        # filename for valid keys
        self.keys_filename = "valid-keys.txt"

        # filename for no quota keys
        self.no_quota_filename = "no-quota-keys.txt"

        # filename for need check again keys
        self.wait_check_filename = "wait-check-keys.txt"

        # filename for extract keys
        self.material_filename = f"material.txt"

        # filename for summary
        self.summary_filename = f"summary.json"

        # filename for links included keys
        self.links_filename = f"links.txt"

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

    def _get_headers(self, token: str, additional: dict = None) -> dict:
        raise NotImplementedError

    def _judge(self, code: int, message: str) -> CheckResult:
        message = trim(message)

        if code == 200 and message:
            return CheckResult.ok()
        elif code == 400:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)
        elif code == 401 or re.findall(r"invalid_api_key", message, flags=re.I):
            return CheckResult.fail(ErrorReason.INVALID_KEY)
        elif code == 402 or re.findall(r"insufficient", message, flags=re.I):
            return CheckResult.fail(ErrorReason.NO_QUOTA)
        elif code == 403 or code == 404:
            return CheckResult.fail(ErrorReason.NO_ACCESS)
        elif code == 418 or code == 429:
            return CheckResult.fail(ErrorReason.RATE_LIMITED)
        elif code >= 500:
            return CheckResult.fail(ErrorReason.SERVER_ERROR)

        return CheckResult.fail(ErrorReason.UNKNOWN)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        url, regex = trim(address), r"^https?://([\w\-_]+\.[\w\-_]+)+"
        if not url and re.match(regex, self.base_url, flags=re.I):
            url = urllib.parse.urljoin(self.base_url, self.completion_path)

        if not re.match(regex, url, flags=re.I):
            logging.error(f"invalid url: {url}, skip to check")
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        headers = self._get_headers(token=token)
        if not headers:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        model = trim(model) or self.default_model
        code, message = chat(url=url, headers=headers, model=model)
        return self._judge(code=code, message=message)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> list[str]:
        raise NotImplementedError


class OpenAILikeProvider(Provider):
    def __init__(
        self,
        name: str,
        base_url: str,
        default_model: str,
        conditions: list[Condition],
        completion_path: str = "",
        model_path: str = "",
        **kwargs,
    ):
        completion_path = trim(completion_path) or DEFAULT_COMPLETION_PATH
        model_path = trim(model_path) or DEFAULT_MODEL_PATH

        super().__init__(name, base_url, completion_path, model_path, default_model, conditions, **kwargs)

    def _get_headers(self, token: str, additional: dict = None) -> dict:
        token = trim(token)
        if not token:
            return None

        if not isinstance(additional, dict):
            additional = {}

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
            "user-agent": USER_AGENT,
        }
        headers.update(additional)

        return headers

    def _judge(self, code: int, message: str) -> CheckResult:
        if code == 200:
            return CheckResult.ok()

        message = trim(message)
        if message:
            if code == 403:
                if re.findall(r"model_not_found", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_MODEL)
                elif re.findall(r"unauthorized", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.INVALID_KEY)
                elif re.findall(r"unsupported_country_region_territory", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_ACCESS)
                elif re.findall(r"exceeded_current_quota_error|insufficient_user_quota|余额不足", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_QUOTA)
            elif code == 429:
                if re.findall(r"insufficient_quota|billing_not_active|欠费|请充值|recharge", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_QUOTA)
                elif re.findall(r"rate_limit_exceeded", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.RATE_LIMITED)

        return super()._judge(code, message)

    def _fetch_models(self, url: str, headers: dict) -> list[str]:
        url = trim(url)
        if not url:
            return []

        content = http_get(url=url, headers=headers, interval=1)
        if not content:
            return []

        try:
            result = json.loads(content)
            return [trim(x.get("id", "")) for x in result.get("data", [])]
        except:
            return []

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> list[str]:
        headers = self._get_headers(token=token)
        if not headers or not self.base_url or not self.model_path:
            return []

        url = urllib.parse.urljoin(self.base_url, self.model_path)
        return self._fetch_models(url=url, headers=headers)


class OpenAIProvider(OpenAILikeProvider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "gpt-4o-mini"
        base_url = "https://api.openai.com"

        super().__init__("openai", base_url, default_model, conditions)


class DoubaoProvider(OpenAILikeProvider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "doubao-pro-32k"
        base_url = "https://ark.cn-beijing.volces.com"

        super().__init__(
            name="doubao",
            base_url=base_url,
            default_model=default_model,
            conditions=conditions,
            completion_path="/api/v3/chat/completions",
            model_path="/api/v3/models",
            model_pattern=r"ep-[0-9]{14}-[a-z0-9]{5}",
        )

    def _judge(self, code: int, message: str) -> CheckResult:
        if code == 404:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super()._judge(code, message)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        model = trim(model)
        if not model:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super().check(token=token, address=address, endpoint=endpoint, model=model)


class QianFanProvider(OpenAILikeProvider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "ernie-4.0-8k-latest"
        base_url = "https://qianfan.baidubce.com"

        super().__init__(
            name="qianfan",
            base_url=base_url,
            default_model=default_model,
            conditions=conditions,
            completion_path="/v2/chat/completions",
            model_path="/v2/models",
            endpoint_pattern=r"[a-z0-9]{8}(?:-[a-z0-9]{4}){3}-[a-z0-9]{12}",
        )

    def _judge(self, code: int, message: str) -> CheckResult:
        if code == 404:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super()._judge(code, message)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        headers = self._get_headers(token=token)
        if not headers:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        endpoint = trim(endpoint)
        if endpoint:
            headers["appid"] = endpoint

        model = trim(model) or self.default_model
        url = urllib.parse.urljoin(self.base_url, self.completion_path)

        code, message = chat(url=url, headers=headers, model=model)
        return self._judge(code=code, message=message)

    def list_models(self, token, address="", endpoint=""):
        headers = self._get_headers(token=token)
        if not headers:
            return []

        endpoint = trim(endpoint)
        if endpoint:
            headers["appid"] = endpoint

        url = urllib.parse.urljoin(self.base_url, self.model_path)
        return self._fetch_models(url=url, headers=headers)


class AnthropicProvider(Provider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "claude-3-5-sonnet-latest"
        super().__init__("anthropic", "https://api.anthropic.com", "/v1/messages", "", default_model, conditions)

    def _get_headers(self, token: str, additional: dict = None) -> dict:
        token = trim(token)
        if not token:
            return None

        return {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
        }

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        token = trim(token)
        if token.startswith("sk-ant-sid01-"):
            url = "https://api.claude.ai/api/organizations"
            headers = {
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "max-age=0",
                "cookie": f"sessionKey={token}",
                "user-agent": USER_AGENT,
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "none",
            }

            # content = http_get(url=url, headers=headers, interval=1)
            content, success = "", False
            attempt, retries, timeout = 0, 3, 10

            req = urllib.request.Request(url, headers=headers, method="GET")
            while attempt < retries:
                try:
                    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as response:
                        content = response.read().decode("utf8")
                        success = True
                        break
                except urllib.error.HTTPError as e:
                    if e.code == 401:
                        return CheckResult.fail(ErrorReason.INVALID_KEY)
                    else:
                        try:
                            content = e.read().decode("utf8")
                            if not content.startswith("{") or not content.endswith("}"):
                                content = e.reason
                        except:
                            content = e.reason

                        if e.code == 403:
                            message = ""
                            try:
                                data = json.loads(content)
                                message = data.get("error", {}).get("message", "")
                            except:
                                message = content

                            if re.findall(r"Invalid authorization", message, flags=re.I):
                                return CheckResult.fail(ErrorReason.INVALID_KEY)

                        if e.code in NO_RETRY_ERROR_CODES:
                            break
                except Exception as e:
                    if not isinstance(e, urllib.error.URLError) or not isinstance(e.reason, socket.timeout):
                        logging.error(f"check claude session error, key: {token}, message: {traceback.format_exc()}")

                attempt += 1
                time.sleep(1)

            if not content or re.findall(r"Invalid authorization", content, flags=re.I):
                return CheckResult.fail(ErrorReason.INVALID_KEY)
            elif not success:
                logging.error(f"check claude session error, key: {token}, message: {content}")
                return CheckResult.fail(ErrorReason.UNKNOWN)

            try:
                data = json.loads(content)
                valid = False
                if data and isinstance(data, list):
                    valid = trim(data[0].get("name", None)) != ""

                    capabilities = data[0].get("capabilities", [])
                    if capabilities and isinstance(capabilities, list) and "claude_pro" in capabilities:
                        logging.info(f"found claude pro key: {token}")

                if not valid:
                    logging.warning(f"check error, anthropic session key: {token}, message: {content}")

                return CheckResult.ok() if valid else CheckResult.fail(ErrorReason.INVALID_KEY)
            except:
                return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super().check(token=token, address=address, endpoint=endpoint, model=model)

    def _judge(self, code: int, message: str) -> CheckResult:
        message = trim(message)
        if re.findall(r"credit balance is too low|Billing|purchase", message, flags=re.I):
            return CheckResult.fail(ErrorReason.NO_QUOTA)
        elif code == 404 and re.findall(r"not_found_error", trim(message), flags=re.I):
            return CheckResult.fail(ErrorReason.NO_MODEL)

        return super()._judge(code, message)

    def list_models(self, token, address: str = "", endpoint: str = "") -> list[str]:
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


class AzureOpenAIProvider(OpenAILikeProvider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "gpt-4o"
        super().__init__(
            name="azure",
            base_url="",
            completion_path="/chat/completions",
            model_path="/models",
            default_model=default_model,
            conditions=conditions,
            address_pattern=r"https://[a-zA-Z0-9_\-\.]+.openai.azure.com/openai/",
        )

        self.api_version = "2024-10-21"

    def _get_headers(self, token: str, additional: dict = None) -> dict:
        token = trim(token)
        if not token:
            return None

        return {
            "accept": "application/json",
            "api-key": token,
            "content-type": "application/json",
            "user-agent": USER_AGENT,
        }

    def _judge(self, code: int, message: str) -> CheckResult:
        if code == 404:
            message = trim(message)
            if re.finditer(r"The API deployment for this resource does not exist", message, flags=re.I):
                return CheckResult.fail(ErrorReason.NO_MODEL)

            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super()._judge(code, message)

    def __generate_address(self, address: str = "", endpoint: str = "", model: str = "") -> str:
        address = trim(address).removesuffix("/")
        if not re.match(r"^https?://([\w\-_]+\.[\w\-_]+)+", address, flags=re.I):
            return ""

        if re.findall(
            r"(xxx|YOUR_RESOURCE_NAME|your_service|YOUR_AZURE_OPENAI_NAME|YOUR-INSTANCE|YOUR_ENDPOINT_NAME|RESOURCE_NAME|YOURAOAIINSTANCE|yourname|YOUR_NAME|YOUR_AOAI_SERVICE|COMPANY|your-deployment-name|YOUR_AOI_SERVICE_NAME|YOUR_AI_ENDPOINT_NAME|YOUR-APP|YOUR-RESOURCE-NAME).openai.azure.com",
            address,
            flags=re.I,
        ):
            return ""

        model = trim(model) or self.default_model
        return f"{address}/deployments/{model}/{self.completion_path}?api-version={self.api_version}"

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        token = trim(token)
        if not token:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        url = self.__generate_address(address=address, endpoint=endpoint, model=model)
        if not url:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super().check(token=token, address=url, endpoint=endpoint, model=model)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> list[str]:
        domain = trim(address).removesuffix("/")
        if not re.match(r"^https?://([\w\-_]+\.[\w\-_]+)+", domain, flags=re.I):
            logging.error(f"invalid domain: {domain}, skip to list models")
            return []

        headers = self._get_headers(token=token)
        if not headers or not self.model_path:
            return []

        url = f"{domain}/{self.model_path}?api-version={self.api_version}"
        return self._fetch_models(url=url, headers=headers)


class GeminiProvider(Provider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "gemini-exp-1206"
        base_url = "https://generativelanguage.googleapis.com"
        sub_path = "/v1beta/models"

        super().__init__("gemini", base_url, sub_path, sub_path, default_model, conditions)

    def _get_headers(self, token: str, additional: dict = None) -> dict:
        return {"accept": "application/json", "content-type": "application/json"}

    def _judge(self, code: int, message: str) -> CheckResult:
        if code == 200:
            return CheckResult.ok()

        message = trim(message)
        if code == 400:
            if re.findall(r"API_KEY_INVALID", message, flags=re.I):
                return CheckResult.fail(ErrorReason.INVALID_KEY)
            elif re.findall(r"FAILED_PRECONDITION", message, flags=re.I):
                return CheckResult.fail(ErrorReason.NO_ACCESS)

        return super()._judge(code, message)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        token = trim(token)
        if not token:
            return False

        model = trim(model) or self.default_model
        url = f"{urllib.parse.urljoin(self.base_url, self.completion_path)}/{model}:generateContent?key={token}"

        params = {"contents": [{"role": "user", "parts": [{"text": DEFAULT_QUESTION}]}]}
        code, message = chat(url=url, headers=self._get_headers(token=token), params=params)
        return self._judge(code=code, message=message)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> list[str]:
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
        regex = r'href="(/[^\s"]+/blob/(?:[^"]+)?)#L\d+"'
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
            if i < page_num:
                time.sleep(random.randint(6, 12))

        links = list(potentials)
    if not links:
        logging.warning(f"[Search] cannot found any link with query: {query}")

    return links


@lru_cache(maxsize=2000)
def collect(
    url: str,
    key_pattern: str,
    retries: int = 3,
    address_pattern: str = "",
    endpoint_pattern: str = "",
    model_pattern: str = "",
) -> list[Service]:
    if not isinstance(url, str) or not isinstance(key_pattern, str):
        return []

    content = http_get(url=url, retries=retries, interval=1)

    # extract keys from content
    key_pattern = trim(key_pattern)
    keys = extract(text=content, regex=key_pattern)
    if not keys:
        return []

    # extract api addresses from content
    address_pattern = trim(address_pattern)
    addresses = extract(text=content, regex=address_pattern)
    if address_pattern and not addresses:
        return []
    if not addresses:
        addresses.append("")

    # extract api endpoints from content
    endpoint_pattern = trim(endpoint_pattern)
    endpoints = extract(text=content, regex=endpoint_pattern)
    if endpoint_pattern and not endpoints:
        return []
    if not endpoints:
        endpoints.append("")

    # extract models from content
    model_pattern = trim(model_pattern)
    models = extract(text=content, regex=model_pattern)
    if model_pattern and not models:
        return []
    if not models:
        models.append("")

    candidates = list()

    # combine keys, addresses and endpoints TODO: optimize this part
    for key, address, endpoint, model in itertools.product(keys, addresses, endpoints, models):
        candidates.append(Service(address=address, endpoint=endpoint, key=key, model=model))

    return candidates


def extract(text: str, regex: str) -> list[str]:
    content, pattern = trim(text), trim(regex)
    if not content or not pattern:
        return []

    items = set()
    try:
        groups = re.findall(pattern, content)
        for x in groups:
            words = list()
            if isinstance(x, str):
                words.append(x)
            elif isinstance(x, (tuple, list)):
                words.extend(list(x))
            else:
                logging.error(f"unknown type: {type(x)}, value: {x}. please optimize your regex")
                continue

            for word in words:
                key = trim(word)
                if key:
                    items.add(key)
    except:
        logging.error(traceback.format_exc())

    return list(items)


def scan(
    session: str,
    provider: Provider,
    with_api: bool = False,
    page_num: int = -1,
    thread_num: int = None,
    fast: bool = False,
    skip: bool = False,
    workspace: str = "",
) -> None:
    def persist(services: list[Service], filepath: str) -> None:
        filepath = trim(filepath)
        if not filepath or not services:
            logging.error("services or filepath cannot be empty")
            return

        lines = [x.serialize() for x in services if x]
        if not write_file(directory=filepath, lines=lines):
            logging.error(f"[Scan] {provider.name}: failed to save keys to file: {filepath}, keys: {lines}")
        else:
            logging.info(f"[Scan] {provider.name}: saved {len(lines)} keys to file: {filepath}")

    if not isinstance(provider, Provider):
        return

    keys_filename = trim(provider.keys_filename)
    if not keys_filename:
        logging.error(f"[Scan] {provider.name}: keys filename cannot be empty")
        return

    workspace = trim(workspace)
    directory = os.path.join(os.path.abspath(workspace) if workspace else PATH, provider.directory)

    valid_keys_file = os.path.join(directory, keys_filename)
    material_keys_file = os.path.join(directory, provider.material_filename)
    links_file = os.path.join(directory, provider.links_filename)

    records: set[Service] = set()
    if os.path.exists(valid_keys_file) and os.path.isfile(valid_keys_file):
        # load exists valid keys
        records.update(load_exist_keys(filepath=valid_keys_file))
        logging.info(f"[Scan] {provider.name}: loaded {len(records)} exists keys from file {valid_keys_file}")

        # backup up exists file with current time
        words = keys_filename.rsplit(".", maxsplit=1)
        keys_filename = f"{words[0]}-{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        if len(words) > 1:
            keys_filename += f".{words[1]}"

        os.rename(valid_keys_file, os.path.join(directory, keys_filename))

    if os.path.exists(material_keys_file) and os.path.isfile(material_keys_file):
        # load potential keys from material file
        records.update(load_exist_keys(filepath=material_keys_file))

    if not skip and provider.conditions:
        candidates: set[Service] = set()
        start_time = time.time()
        for condition in provider.conditions:
            if not isinstance(condition, Condition):
                continue

            query, regex = condition.query, condition.regex
            logging.info(
                f"[Scan] {provider.name}: start to search new keys with query: {query or regex}, regex: {regex}"
            )

            parts = recall(
                regex=regex,
                session=session,
                query=query,
                with_api=with_api,
                page_num=page_num,
                thread_num=thread_num,
                fast=fast,
                links_file=links_file,
                extra_params=provider.extras,
            )

            if parts:
                candidates.update(parts)

        # merge new keys with exists keys
        records.update(candidates)

        cost = time.time() - start_time
        count, total = len(provider.conditions), len(candidates)
        logging.info(f"[Scan] {provider.name}: cost {cost:.2f}s to search {count} conditions, got {total} new keys")

    if not records:
        logging.warning(f"[Scan] {provider.name}: cannot extract any candidate with conditions: {provider.conditions}")
        return

    items, caches = [], []
    for record in records:
        if not isinstance(record, Service) or not record.key:
            continue

        items.append([record.key, record.address, record.endpoint, record.model])
        caches.append(record)

    logging.info(f"[Scan] {provider.name}: start to verify {len(items)} potential keys")
    masks: list[CheckResult] = multi_thread_run(func=provider.check, tasks=items, thread_num=thread_num)

    # remove invalid keys and ave all potential keys to material file
    materials: list[Service] = [caches[i] for i in range(len(masks)) if masks[i].reason != ErrorReason.INVALID_KEY]
    if materials:
        persist(services=materials, filepath=material_keys_file)

    # save candidates and avaiable status
    statistics = dict()

    # can be used keys
    valid_services: list[Service] = [caches[i] for i in range(len(masks)) if masks[i].available]
    if not valid_services:
        logging.warning(f"[Scan] {provider.name}: cannot found any key with conditions: {provider.conditions}")
    else:
        persist(services=valid_services, filepath=valid_keys_file)

        statistics.update({s: True for s in valid_services})

    # no quota keys
    no_quota_services: list[Service] = [
        caches[i] for i in range(len(masks)) if not masks[i].available and masks[i].reason == ErrorReason.NO_QUOTA
    ]
    if no_quota_services:
        statistics.update({s: False for s in no_quota_services})

        # save no quota keys to file
        no_quota_keys_file = os.path.join(directory, provider.no_quota_filename)
        persist(services=no_quota_services, filepath=no_quota_keys_file)

    # not expired keys but wait to check again keys
    wait_check_services: list[Service] = [
        caches[i]
        for i in range(len(masks))
        if not masks[i].available
        and masks[i].reason in [ErrorReason.RATE_LIMITED, ErrorReason.NO_MODEL, ErrorReason.UNKNOWN]
    ]
    if wait_check_services:
        statistics.update({s: False for s in wait_check_services})

        # save wait check keys to file
        wait_check_keys_file = os.path.join(directory, provider.wait_check_filename)
        persist(services=wait_check_services, filepath=wait_check_keys_file)

    # list supported models for each key
    services, tasks = [], []
    for service in statistics.keys():
        tasks.append([service.key, service.address, service.endpoint])
        services.append(service)

    if not tasks:
        logging.error(f"[Scan] {provider.name}: no keys to list models")
        return

    data = dict()
    models = multi_thread_run(func=provider.list_models, tasks=tasks, thread_num=thread_num)

    for i in range(len(tasks)):
        service: Service = services[i]
        item = {
            "available": statistics.get(service),
            "models": (models[i] if models else []) or [],
        }

        if service.address:
            item["address"] = service.address
        if service.endpoint:
            item["endpoint"] = service.endpoint

        data[service.key] = item

    summary_path = os.path.join(directory, provider.summary_filename)
    if write_file(directory=summary_path, lines=json.dumps(data, ensure_ascii=False, indent=4)):
        logging.info(f"[Scan] {provider.name}: saved {len(services)} keys summary to file: {summary_path}")
    else:
        logging.error(f"[Scan] {provider.name}: failed to save keys summary to file: {summary_path}, data: {data}")


def load_exist_keys(filepath: str) -> set[Service]:
    lines = read_file(filepath=filepath)
    if not lines:
        return set()

    return set([Service.deserialize(text=line) for line in lines])


def recall(
    regex: str,
    session: str,
    query: str = "",
    with_api: bool = False,
    page_num: int = -1,
    thread_num: int = None,
    fast: bool = False,
    links_file: str = "",
    extra_params: dict = None,
) -> list[Service]:
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
    query = trim(query)
    if not query:
        text = regex.replace("/", "\\/")
        query = f"/{text}/"

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

    if not isinstance(extra_params, dict):
        extra_params = dict()

    address_pattern = extra_params.get("address_pattern", "")
    endpoint_pattern = extra_params.get("endpoint_pattern", "")
    model_pattern = extra_params.get("model_pattern", "")

    tasks = [[x, regex, 3, address_pattern, endpoint_pattern, model_pattern] for x in links if x]
    result = multi_thread_run(func=collect, tasks=tasks, thread_num=thread_num)

    return list(itertools.chain.from_iterable(result))


def chat(
    url: str, headers: dict, model: str = "", params: dict = None, retries: int = 2, timeout: int = 10
) -> tuple[int, str]:
    def output(code: int, message: str, debug: bool = False) -> None:
        text = f"[Chat] failed to request url: {url}, headers: {headers}, status code: {code}, message: {message}"
        if debug:
            logging.debug(text)
        else:
            logging.error(text)

    url, model = trim(url), trim(model)
    if not url:
        logging.error(f"[Chat] url cannot be empty")
        return 400, None

    if not isinstance(headers, dict):
        logging.error(f"[Chat] headers must be a dict")
        return 400, None
    elif len(headers) == 0:
        headers["content-type"] = "application/json"

    if not params or not isinstance(params, dict):
        if not model:
            logging.error(f"[Chat] model cannot be empty")
            return 400, None

        params = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": DEFAULT_QUESTION}],
        }

    payload = json.dumps(params).encode("utf8")
    timeout = max(1, timeout)
    retries = max(1, retries)
    code, message, attempt = 400, None, 0

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    while attempt < retries:
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as response:
                code = 200
                message = response.read().decode("utf8")
                break
        except urllib.error.HTTPError as e:
            code = e.code
            if code != 401:
                try:
                    # read response body
                    message = e.read().decode("utf8")

                    # not a json string, use reason instead
                    if not message.startswith("{") or not message.endswith("}"):
                        message = e.reason
                except:
                    message = e.reason

                # print http status code and error message
                output(code=code, message=message, debug=False)

            if code in NO_RETRY_ERROR_CODES:
                break
        except Exception:
            output(code=code, message=traceback.format_exc(), debug=True)

        attempt += 1
        time.sleep(1)

    return code, message


def scan_anthropic_keys(
    session: str,
    with_api: bool = False,
    page_num: int = -1,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
) -> None:
    regex = r"sk-ant-(?:sid01|api03)-[a-zA-Z0-9_\-]{93}AA"
    if with_api:
        conditions = [Condition(query="sk-ant-api03-", regex=regex), Condition(query="sk-ant-sid01-", regex=regex)]
    else:
        conditions = [Condition(query="", regex=regex)]

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
        workspace=workspace,
    )


def scan_azure_keys(
    session: str,
    with_api: bool = False,
    page_num: int = -1,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
) -> None:
    # TODO: optimize query syntax for github api
    query = "/https:\/\/[a-zA-Z0-9_\-\.]+.openai.azure.com\/openai\// AND /(?-i)[a-z0-9]{32}/"
    if with_api:
        query = "openai.azure.com/openai/deployments"

    regex = r"[a-z0-9]{32}"
    conditions = [Condition(query=query, regex=regex)]
    default_model = "gpt-4o"
    provider = AzureOpenAIProvider(conditions=conditions, default_model=default_model)

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        page_num=page_num,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
        workspace=workspace,
    )


def scan_gemini_keys(
    session: str,
    with_api: bool = False,
    page_num: int = -1,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
) -> None:
    query = '/AIzaSy[a-zA-Z0-9_\-]{33}/ AND content:"gemini"'
    if with_api:
        query = '"AIzaSy" AND "gemini"'

    regex = r"AIzaSy[a-zA-Z0-9_\-]{33}"
    conditions = [Condition(query=query, regex=regex)]
    default_model = "gemini-exp-1206"
    provider = GeminiProvider(conditions=conditions, default_model=default_model)

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        page_num=page_num,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
        workspace=workspace,
    )


def scan_openai_keys(
    session: str,
    with_api: bool = False,
    page_num: int = -1,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
) -> None:
    # TODO: optimize query syntax for github api
    query = '"T3BlbkFJ"' if with_api else ""
    regex = r"sk(?:-proj)?-[a-zA-Z0-9]{20}T3BlbkFJ[a-zA-Z0-9]{20}|sk-proj-(?:[a-zA-Z0-9_\-]{91}|[a-zA-Z0-9_\-]{123}|[a-zA-Z0-9_\-]{155})A|sk-svcacct-[A-Za-z0-9_\-]+T3BlbkFJ[A-Za-z0-9_\-]+"

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
        workspace=workspace,
    )


def scan_doubao_keys(
    session: str,
    with_api: bool = False,
    page_num: int = -1,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
) -> None:
    # TODO: optimize query syntax for github api
    query = '"https://ark.cn-beijing.volces.com" AND /[a-z0-9]{8}(?:-[a-z0-9]{4}){3}-[a-z0-9]{12}/'
    if with_api:
        query = '"https://ark.cn-beijing.volces.com" AND "ep-"'

    regex = r"[a-z0-9]{8}(?:-[a-z0-9]{4}){3}-[a-z0-9]{12}"
    conditions = [Condition(query=query, regex=regex)]
    provider = DoubaoProvider(conditions=conditions, default_model="doubao-pro-32k")

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        page_num=page_num,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
        workspace=workspace,
    )


def scan_qianfan_keys(
    session: str,
    with_api: bool = False,
    page_num: int = -1,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
) -> None:
    query = "bce-v3/ALTAK-" if with_api else "/bce-v3\/ALTAK-[a-zA-Z0-9]{21}\/[a-z0-9]{40}/"
    regex = r"bce-v3/ALTAK-[a-zA-Z0-9]{21}/[a-z0-9]{40}"
    conditions = [Condition(query=query, regex=regex)]
    provider = QianFanProvider(conditions=conditions, default_model="ernie-4.0-8k-latest")

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        page_num=page_num,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
        workspace=workspace,
    )


def scan_others(args: argparse.Namespace) -> None:
    if not args or not isinstance(args, argparse.Namespace):
        return

    name = trim(args.pn)
    if not name:
        logging.error(f"provider name cannot be empty")
        return

    default_model = trim(args.pm)
    if not default_model:
        logging.error(f"model name as default cannot be empty")
        return

    base_url = trim(args.pb)
    if not re.match(r"^https?://([\w\-_]+\.[\w\-_]+)+", base_url):
        logging.error(f"invalid base url: {base_url}")
        return

    pattern = trim(args.pp)
    if not pattern:
        logging.error(f"pattern for extracting keys cannot be empty")
        return

    query = trim(args.pq)
    if args.rest and not query:
        logging.error(f"query cannot be empty when using rest api")
        return

    conditions = [Condition(query=query, regex=pattern)]
    provider = OpenAILikeProvider(
        name=name,
        base_url=base_url,
        default_model=default_model,
        conditions=conditions,
        completion_path=args.pc,
        model_path=args.pl,
    )

    return scan(
        session=args.session,
        provider=provider,
        with_api=args.rest,
        page_num=args.num,
        thread_num=args.thread,
        fast=args.fast,
        skip=args.elide,
        workspace=args.workspace,
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
        message = e.read()
        try:
            message = str(message, encoding="utf8")
        except:
            message = gzip.decompress(message).decode("utf8")

        if not (message.startswith("{") and message.endswith("}")):
            message = e.reason

        logging.error(f"failed to request url: {url}, status code: {e.code}, message: {message}")

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


def main(args: argparse.Namespace) -> None:
    session = trim(args.session)

    if args.azure:
        scan_azure_keys(
            session=session,
            with_api=args.rest,
            page_num=args.num,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
        )

    if args.doubao:
        scan_doubao_keys(
            session=session,
            with_api=args.rest,
            page_num=args.num,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
        )

    if args.claude:
        scan_anthropic_keys(
            session=session,
            with_api=args.rest,
            page_num=args.num,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
        )

    if args.gemini:
        scan_gemini_keys(
            session=session,
            with_api=args.rest,
            page_num=args.num,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
        )

    if args.openai:
        scan_openai_keys(
            session=session,
            with_api=args.rest,
            page_num=args.num,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
        )

    if args.qianfan:
        scan_qianfan_keys(
            session=session,
            with_api=args.rest,
            page_num=args.num,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
        )

    if args.variant:
        return scan_others(args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--azure",
        dest="azure",
        action="store_true",
        default=False,
        help="scan azure's openai api keys",
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
        "-d",
        "--doubao",
        dest="doubao",
        action="store_true",
        default=False,
        help="scan doubao api keys",
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
        "-q",
        "--qianfan",
        dest="qianfan",
        action="store_true",
        default=False,
        help="scan qianfan api keys",
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

    parser.add_argument(
        "-t",
        "--thread",
        type=int,
        required=False,
        default=-1,
        help="concurrent thread number. default is -1, mean auto select",
    )

    parser.add_argument(
        "-v",
        "--variant",
        dest="variant",
        action="store_true",
        default=False,
        help="scan other api keys like openai",
    )

    parser.add_argument(
        "-w",
        "--workspace",
        type=str,
        default=PATH,
        required=False,
        help="workspace path",
    )

    parser.add_argument(
        "-pb",
        "--provider-base",
        dest="pb",
        type=str,
        default="",
        required=False,
        help="base url, must be a valid url start with 'http://' or 'https://'",
    )

    parser.add_argument(
        "-pc",
        "--provider-chat",
        dest="pc",
        type=str,
        default="",
        required=False,
        help="chat api path, default is '/v1/chat/completions'",
    )

    parser.add_argument(
        "-pl",
        "--provider-list",
        dest="pl",
        type=str,
        default="",
        required=False,
        help="list models api path, default is '/v1/models'",
    )

    parser.add_argument(
        "-pm",
        "--provider-model",
        dest="pm",
        type=str,
        default="",
        required=False,
        help="default model name",
    )

    parser.add_argument(
        "-pn",
        "--provider-name",
        dest="pn",
        type=str,
        default="",
        required=False,
        help="provider name, contain only letters, numbers, '_' and '-'",
    )

    parser.add_argument(
        "-pq",
        "--provider-query",
        dest="pq",
        type=str,
        default="",
        required=False,
        help="query syntax for github search",
    )

    parser.add_argument(
        "-pp",
        "--provider-pattern",
        dest="pp",
        type=str,
        default="",
        required=False,
        help="pattern for extract keys from code, default is 'sk-[a-zA-Z0-9_\-]{48}'",
    )

    main(parser.parse_args())

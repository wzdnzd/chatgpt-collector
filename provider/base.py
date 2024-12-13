# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-08-12

import json
import os
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from random import choice

import urlvalidator
import utils
from logger import logger

EMAILS_DOMAINS = [
    "gmail.com",
    "outlook.com",
    "163.com",
    "126.com",
    "sina.com",
    "hotmail.com",
    "qq.com",
    "foxmail.com",
    "hotmail.com",
    "yahoo.com",
]


class APIStyle(Enum):
    OPENAI = 1
    AZURE = 2
    OTHER = 3

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def is_standard(style: str) -> bool:
        return style.lower() in ["openai", "azure"]

    @staticmethod
    def from_str(style: str):
        style = utils.trim(style).lower()
        if style == "openai":
            return APIStyle.OPENAI
        elif style == "azure":
            return APIStyle.AZURE
        else:
            return APIStyle.OTHER


@dataclass
class Model(object):
    # model name
    name: str

    # model type
    ownedby: str = ""


@dataclass
class Account(object):
    # domain address
    domain: str

    # email
    email: str

    # password
    password: str

    # username
    username: str

    # access token
    token: str = ""

    # token type
    token_type: str = "Bearer"

    # cookie
    cookie: str = ""

    # whether account is available
    available: bool = True

    # available endpoints
    endpoints: list[str] = field(default_factory=list)

    # quota
    quota: float = -1


@dataclass
class ServiceInfo(Account):
    # api address
    api_urls: list[str] = field(default_factory=list)

    # api style
    style: APIStyle = APIStyle.OPENAI

    # api keys can be used
    api_keys: list[str] = field(default_factory=list)

    # models can be used
    models: list[Model] = field(default_factory=list)

    # sepcial headers for service
    headers: dict = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        account: Account,
        api_urls: str,
        api_keys: list[str],
        models: list[Model],
        style: APIStyle = APIStyle.OPENAI,
        headers: dict = None,
    ):
        if isinstance(api_urls, str):
            api_urls = [api_urls]
        elif not isinstance(api_urls, list):
            api_urls = []

        return cls(
            domain=account.domain,
            email=account.email,
            password=account.password,
            username=account.username,
            token=account.token,
            token_type=account.token_type,
            cookie=account.cookie,
            available=account.available,
            endpoints=account.endpoints,
            quota=account.quota,
            api_urls=api_urls,
            style=style,
            api_keys=api_keys,
            models=models,
            headers=headers,
        )

    def to_dict(self) -> dict:
        data = {
            "domain": self.domain,
            "email": self.email,
            "password": self.password,
            "username": self.username,
            "available": self.available,
            "token": self.token,
            "token_type": self.token_type,
            "api_url": self.api_urls,
            "style": str(self.style),
            "api_keys": self.api_keys,
            "models": [asdict(model) for model in self.models],
        }

        if self.quota >= 0:
            data["quota"] = self.quota
        if self.endpoints:
            data["endpoints"] = self.endpoints

        return data

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def deserialize(cls, content: str):
        try:
            data = json.loads(content)
            domain = data.get("domain", "")
            email = data.get("email", "")
            password = data.get("password", "")
            username = data.get("username", "")
            available = data.get("available", True)
            token = data.get("token", "")
            token_type = data.get("token_type", "Bearer")
            api_urls = data.get("api_url", [])
            style = data.get("style", "openai")
            api_keys = data.get("api_keys", [])
            models = [Model(**model) for model in data.get("models", [])]
            quota = data.get("quota", -1)
            endpoints = data.get("endpoints", [])

            return cls(
                domain=domain,
                email=email,
                password=password,
                username=username,
                token=token,
                token_type=token_type,
                available=available,
                quota=float(quota) if utils.is_number(quota) else -1,
                api_urls=api_urls,
                style=APIStyle.from_str(style),
                api_keys=api_keys,
                models=models,
                endpoints=endpoints,
            )
        except:
            logger.error(f"deserialize service information failed, content: {content}")
            return None


class ServiceProvider(object):
    def __init__(self, domain: str):
        domain = utils.trim(domain)
        if not urlvalidator.isurl(url=domain):
            raise ValueError(f"invalid domain: {domain}")

        self.domain = domain.removesuffix("/")

        # model api path
        self._model_api = f"{self.domain}/api/models"

        # default headers
        self.headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip",
            "Content-Type": "application/json",
            "Host": utils.extract_domain(url=domain, include_protocal=False),
            "Origin": domain,
            "Referer": domain + "/",
            "User-Agent": utils.USER_AGENT,
        }

    def _generate_username(self, username: str = "", default: str = "") -> str:
        username = utils.trim(username)
        if not username:
            username = utils.trim(default) or utils.random_chars(8, punctuation=False)

        return username

    def _generate_password(self, password: str = "", default: str = "", punctuation: bool = True) -> str:
        password = utils.trim(password)
        if not password:
            password = utils.trim(default) or utils.random_chars(16, punctuation=punctuation)

        return password

    def _generate_email(self, email: str = "", username: str = "", default: str = "") -> str:
        email = utils.trim(email)
        regex = r"^\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*$"

        if not re.match(regex, email):
            default = utils.trim(default)
            if re.match(regex, default):
                email = default
            else:
                username = self._generate_username(username).lower()
                email = f"{username}@{choice(EMAILS_DOMAINS)}"

        return email

    def _get_api_urls(self, models: list[Model] = None) -> list[str]:
        return [f"{self.domain}/v1/chat/completions"]

    def _get_headers(
        self,
        key_name: str = "Authorization",
        token: str = "",
        token_type: str = "Bearer",
        cookie: str = "",
        extra: dict = None,
    ) -> dict:
        token = utils.trim(token)
        token_type = utils.trim(token_type)
        key_name = utils.trim(key_name) or "Authorization"

        headers = deepcopy(self.headers)
        if token:
            content = f"{token_type} {token}" if token_type else token
            headers[key_name] = content

        cookie = utils.trim(cookie)
        if cookie:
            headers["Cookie"] = cookie

        if extra and isinstance(extra, dict):
            headers.update(extra)

        return headers

    def _register(self, username: str, password: str, email: str, **kwargs) -> Account:
        raise NotImplementedError

    def _get_account_by_token(self, token: str, key_name: str = "Authorization", token_type: str = "Bearer") -> Account:
        raise NotImplementedError

    def _get_account(
        self, token: str = "", username: str = "", password: str = "", email: str = "", **kwargs
    ) -> Account:
        account, token = None, utils.trim(token)
        if token:
            key_name = kwargs.get("key_name", "Authorization")

            # check access token validity and get account info first
            account = self._get_account_by_token(
                token=token,
                key_name=key_name,
                token_type=kwargs.get("token_type", "Bearer"),
            )

            account.password = account.password or utils.trim(password)

        if not account:
            account = self._register(username=username, password=password, email=email, **kwargs)
        else:
            account.password = password

        return account

    def _get_api_keys(self, token: str, token_type: str = "Bearer", cookie: str = "") -> list[str]:
        raise NotImplementedError

    @classmethod
    def api_style(cls) -> APIStyle:
        return APIStyle.OPENAI

    def _construct_models(self, data: dict) -> list[Model]:
        items = data.get("data", [])
        if not items or not isinstance(items, list):
            logger.error(f"[{self.__class__.__name__}] fetch models failed, domain: {self.domain}, data: {data}")
            return []

        models = list()
        for item in items:
            if not item or not isinstance(item, dict):
                continue

            name = item.get("id", "") or item.get("name", "")
            ownedby = item.get("owned_by", "")
            models.append(Model(name=name, ownedby=ownedby))

        return models

    def _get_models(
        self, token: str, token_type: str = "Bearer", cookie: str = "", endpoints: list[str] = None
    ) -> list[Model]:
        token = utils.trim(token)
        cookie = utils.trim(cookie)

        if not token and not cookie:
            logger.error(
                f"[{self.__class__.__name__}] cannot list models due to token and cookie is empty, domain: {self.domain}"
            )
            return []

        headers = self._get_headers(token=token, token_type=token_type, cookie=cookie)
        content = utils.http_get(url=self._model_api, headers=headers, interval=2)
        try:
            data = json.loads(content)
            return [] if not data or not isinstance(data, (dict, list)) else self._construct_models(data)
        except:
            logger.error(f"[{self.__class__.__name__}] cannot list models, domain: {self.domain}")
            return []

    @classmethod
    def _name(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def _get_default_persist_file(cls) -> str:
        return os.path.join(utils.PATH, "data", cls._name(), "accounts.txt")

    def _key_as_token(self) -> bool:
        return False

    def get_service(
        self, token: str = "", username: str = "", password: str = "", email: str = "", **kwargs
    ) -> ServiceInfo:
        account = self._get_account(token=token, username=username, password=password, email=email, **kwargs)
        if not account:
            logger.error(f"[{self.__class__.__name__}] cannot get any account, domain: {self.domain}")
            return None

        style = self.api_style()
        api_keys = self._get_api_keys(token=account.token, token_type=account.token_type, cookie=account.cookie)

        token = account.token
        if api_keys and self._key_as_token():
            for key in api_keys:
                if not key:
                    continue

                models = self._get_models(
                    token=key,
                    token_type=account.token_type,
                    cookie=account.cookie,
                    endpoints=account.endpoints,
                )
                if models:
                    token = key
                    break
        else:
            models = self._get_models(
                token=token,
                token_type=account.token_type,
                cookie=account.cookie,
                endpoints=account.endpoints,
            )

        api_url = self._get_api_urls(models=models)
        headers = self._get_headers(token=token, token_type=account.token_type, cookie=account.cookie)

        service = ServiceInfo.new(
            account=account,
            api_urls=api_url,
            api_keys=api_keys,
            models=models,
            style=style,
            headers=headers,
        )

        serialize = kwargs.get("serialize", True)
        if serialize:
            filename = utils.trim(kwargs.get("filename", ""))
            if not filename:
                filename = self._get_default_persist_file()

            content = service.serialize()
            success = utils.write_file(filename=filename, lines=content, overwrite=False)
            if not success:
                logger.error(
                    f"[{self.__class__.__name__}] write service information failed, filename: {filename}, content: {content}"
                )

        return service

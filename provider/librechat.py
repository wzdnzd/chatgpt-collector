# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-08-13

import json
from typing import Iterable

import utils
from logger import logger
from provider.base import Account, APIStyle, Model
from provider.openwebui import OpenWebUI


class LibreChat(OpenWebUI):
    def __init__(self, domain: str):
        super().__init__(domain)

        # login url
        self._login_url = f"{self.domain}/api/auth/login"

        # register url
        self._register_url = f"{self.domain}/api/auth/register"

    def _get_account_by_token(self, token: str, key_name: str = "Authorization", token_type: str = "Bearer") -> Account:
        token = utils.trim(token)
        if not token:
            logger.error(f"[LibreChat] cannot get account due to token is empty, domain: {self.domain}")
            return None

        url = f"{self.domain}/api/user"
        headers = self._get_headers(key_name=key_name, token=token, token_type=token_type)
        content = utils.http_get(url=url, headers=headers, interval=1)

        try:
            data = json.loads(content)
            username = data.get("username", "")
            email = data.get("email", "")
            return Account(domain=self.domain, username=username, password="", email=email, token=token)
        except:
            logger.error(f"[LibreChat] cannot get account with token, domain: {self.domain}")
            return None

    def _get_register_body(self, username: str, password: str, email: str, **kwargs) -> dict:
        return {
            "name": username,
            "username": username,
            "email": email,
            "password": password,
            "confirm_password": password,
        }

    def _construct_account(self, data: dict, **kwargs) -> Account:
        token, cookie = data.get("token", ""), ""
        headers = kwargs.get("headers", {})
        if headers:
            cookie = headers.get("Set-Cookie", "")
            if not token:
                token = headers.get("Authorization", "").removeprefix("Bearer ")

        user = data.get("user", {})
        if not token or not user or not isinstance(user, dict):
            logger.error(f"[LibreChat] cannot extract user from response, domain: {self.domain}")
            return None

        username = user.get("name", "") or kwargs.get("username", "") or user.get("username", "")
        email = user.get("email", "") or kwargs.get("email", "")
        password = kwargs.get("password", "")

        return Account(
            domain=self.domain,
            username=username,
            password=password,
            email=email,
            token=token,
            cookie=cookie,
        )

    def _get_api_urls(self, models: list[Model] = None) -> list[str]:
        if not models:
            logger.error(f"[LibreChat] cannot get api urls due to models is empty, domain: {self.domain}")
            return []

        # api fromat: https://xxx.com/api/ask/{ownedby}
        urls = set()
        for model in models:
            if not model or not isinstance(model, Model):
                continue

            urls.add(f"{self.domain}/api/ask/{model.ownedby.lower()}")

        return list(urls)

    @classmethod
    def api_style(cls) -> APIStyle:
        return APIStyle.OTHER

    def _construct_models(self, data: dict) -> list[Model]:
        models = list()
        for k, v in data.items():
            if not k or not v or not isinstance(v, list):
                continue

            for name in v:
                models.append(Model(name=name, ownedby=k))

        return models

    def _get_api_keys(self, token: str, token_type: str = "Bearer", cookie: str = "") -> list[str]:
        logger.warning(f"[LibreChat] not support to create api key, domain: {self.domain}")
        return []

    def _get_models(
        self, token: str, token_type: str = "Bearer", cookie: str = "", endpoints: list[str] = None
    ) -> list[Model]:
        models = super()._get_models(token, token_type, cookie, endpoints)
        if endpoints and isinstance(endpoints, Iterable):
            models = [model for model in models if model.ownedby in endpoints]

        return models

    def _get_available_endpoints(self, headers: dict) -> list[str]:
        if not headers:
            return []

        url = f"{self.domain}/api/endpoints"
        content = utils.http_get(url=url, headers=headers, interval=1)
        try:
            data = json.loads(content)
            if not data or not isinstance(data, dict):
                logger.error(f"[LibreChat] cannot get available endpoints, domain: {self.domain}")
                return []

            endpoints = list()
            for k, v in data.items():
                if not k or not v or not isinstance(v, dict):
                    continue

                if not v.get("userProvide", True):
                    endpoints.append(k)

            return endpoints
        except:
            logger.error(f"[LibreChat] cannot get available endpoints, domain: {self.domain}")
            return []

    def _do_check(self, data: dict) -> bool:
        return data.get("registrationEnabled", False) and data.get("emailLoginEnabled", False)

    def _get_account(
        self, token: str = "", username: str = "", password: str = "", email: str = "", **kwargs
    ) -> Account:
        account = super()._get_account(token, username, password, email, **kwargs)
        if account:
            headers = self._get_headers(token=account.token, cookie=account.cookie)
            endpoints = self._get_available_endpoints(headers=headers)
            if not endpoints:
                logger.error(f"[LibreChat] no available endpoints to use, domain: {self.domain}")
                account = None
            else:
                account.endpoints = endpoints

        return account

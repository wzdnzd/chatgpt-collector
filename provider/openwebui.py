# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-08-12


import json

import utils
from logger import logger
from provider.base import Account, Model, ServiceProvider


class OpenWebUI(ServiceProvider):
    def __init__(self, domain: str):
        super().__init__(domain)

        # login url
        self._login_url = f"{self.domain}/api/v1/auths/signin"

        # register url
        self._register_url = f"{self.domain}/api/v1/auths/signup"

    def _register(self, username: str, password: str, email: str, **kwargs) -> Account:
        password = self._generate_password(password, punctuation=False)
        email = self._generate_email(email, username)

        # check if account already exists
        account = self._login(email=email, password=password)
        if account:
            logger.info(f"[{self.__class__.__name__}] account already exists, domain: {self.domain}, email: {email}")
            return account

        if not self._check():
            logger.error(f"[{self.__class__.__name__}] signup is disabled, domain: {self.domain}")
            return None

        username = self._generate_username(username)
        payload = self._get_register_body(username=username, password=password, email=email, **kwargs)
        if not payload or not isinstance(payload, dict):
            logger.error(f"[{self.__class__.__name__}] invalid register body, domain: {self.domain}")
            return None

        # register a account
        response = utils.http_post_noerror(url=self._register_url, headers=self.headers, params=payload, timeout=15)
        data = utils.read_response(response=response, deserialize=True)
        if not data or not isinstance(data, dict):
            logger.error(f"[{self.__class__.__name__}] signup failed, domain: {self.domain}")
            return None

        return self._construct_account(
            data,
            username=username,
            email=email,
            password=password,
            headers=response.headers,
        )

    def _get_register_body(self, username: str, password: str, email: str, **kwargs) -> dict:
        payload = {"name": username, "password": password, "email": email}
        if kwargs:
            payload.update(kwargs)

        return payload

    def _get_account_by_token(self, token: str, key_name: str = "Authorization", token_type: str = "Bearer") -> Account:
        token = utils.trim(token)
        if not token:
            logger.error(f"[OpenWebUI] cannot get account due to token is empty, domain: {self.domain}")
            return None

        url = f"{self.domain}/api/v1/auths/"
        headers = self._get_headers(key_name=key_name, token=token, token_type=token_type)
        content = utils.http_get(url=url, headers=headers, interval=1)

        try:
            data = json.loads(content)
            username = data.get("name", "")
            email = data.get("email", "")
            return Account(
                domain=self.domain,
                username=username,
                password="",
                email=email,
                token=token,
                token_type=token_type,
            )
        except:
            logger.error(f"[OpenWebUI] cannot get account with token, domain: {self.domain}")
            return None

    def _generate_login_body(self, email: str, password: str) -> dict:
        email = utils.trim(email)
        password = utils.trim(password)

        if not email or not password:
            return {}

        return {"email": email, "password": password}

    def _construct_account(self, data: dict, **kwargs) -> Account:
        available = data.get("role", "") == "user"
        if not available:
            logger.error(f"[{self.__class__.__name__}] signup failed, pending to activate, domain: {self.domain}")
            return None

        token = data.get("token", "")
        token_type = data.get("token_type", "Bearer")

        password = kwargs.get("password", "")
        email = data.get("email", "") or kwargs.get("email", "")
        username = data.get("name", "") or kwargs.get("username", "")

        return Account(
            domain=self.domain,
            username=username,
            password=password,
            email=email,
            token=token,
            token_type=token_type,
        )

    def _login(self, email: str, password: str) -> Account:
        email = utils.trim(email)
        password = utils.trim(password)

        if not email or not password:
            logger.error(f"[{self.__class__.__name__}] login failed, email or password is empty, domain: {self.domain}")
            return None

        payload = self._generate_login_body(email=email, password=password)
        if not payload or not isinstance(payload, dict):
            logger.error(f"[{self.__class__.__name__}] invalid login body, domain: {self.domain}")
            return None

        response = utils.http_post_noerror(url=self._login_url, headers=self.headers, params=payload, timeout=15)
        data = utils.read_response(response=response, expected=200, deserialize=True)
        if not data or not isinstance(data, dict):
            logger.error(f"[{self.__class__.__name__}] login failed, domain: {self.domain}, email: {email}")
            return None

        return self._construct_account(data, email=email, password=password)

    def _get_api_urls(self, models: list[Model] = None) -> list[str]:
        return [f"{self.domain}/openai/chat/completions"]

    def _get_api_keys(self, token: str, token_type: str = "Bearer", cookie: str = "") -> list[str]:
        token = utils.trim(token)
        if not token:
            logger.error(f"[OpenWebUI] cannot create api key due to token is empty, domain: {self.domain}")
            return []

        url = f"{self.domain}/api/v1/auths/api_key"
        headers = self._get_headers(token=token, token_type=token_type, cookie=cookie)
        content = utils.http_get(url=url, headers=headers, interval=1)

        apikey = ""
        try:
            data = json.loads(content)
            apikey = data.get("api_key", "")
        except:
            pass

        if not apikey:
            response = utils.http_post_noerror(url=url, headers=headers, timeout=15)
            apikey = utils.read_response(response=response, expected=200, deserialize=True, key="api_key")

        return [apikey] if apikey else []

    def _check(self) -> bool:
        url = f"{self.domain}/api/config"
        content = utils.http_get(url=url, headers=self.headers, interval=2)
        if not content:
            return False

        try:
            data = json.loads(content)
            return False if not data or not isinstance(data, dict) else self._do_check(data)
        except:
            logger.error(f"[{self.__class__.__name__}] load config failed, domain: {self.domain}")
            return False

    def _do_check(self, data: dict) -> bool:
        features = data.get("features", {})
        if not isinstance(features, dict):
            return False

        return features.get("enable_signup", True)

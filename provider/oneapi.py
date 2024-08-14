# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-08-12


import json

import utils
from logger import logger
from provider.base import Account, ServiceProvider


class OneAPI(ServiceProvider):
    def __init__(self, domain: str):
        super().__init__(domain)

        # model api url
        self._model_api = f"{self.domain}/v1/models"

    def _register(self, username: str, password: str, email: str, **kwargs) -> Account:
        username = self._generate_username(username, default="root")
        password = self._generate_password(password, default="123456")
        email = self._generate_email(email)

        # check if account already exists
        account = self._login(username=username, password=password)
        if account:
            logger.info(f"[OneAPI] account already exists, domain: {self.domain}, username: {username}")
            return account

        url = f"{self.domain}/api/user/register?turnstile="
        if username == "root":
            username = self._generate_username()
            email = self._generate_email(email=email, username=username)

            logger.warning(f"[OneAPI] username: root has been used, replace with {username}, domain: {self.domain}")

        if len(password) < 8:
            password = self._generate_password(punctuation=True)

        # TODO: support temporary email to receive activation email and activate account
        payload = {
            "username": username,
            "password": password,
            "password2": password,
            "email": email,
            "verification_code": "",
        }

        # register a account
        response = utils.http_post_noerror(url=url, headers=self.headers, params=payload, timeout=15)
        data = utils.read_response(response=response, deserialize=True)
        if not data or not isinstance(data, dict) or not data.get("success", False):
            logger.error(f"[OneAPI] signup failed, domain: {self.domain}")
            return None

        return self._login(username=username, password=password)

    def _get_account_by_token(self, token: str, key_name: str = "Authorization", token_type: str = "Bearer") -> Account:
        token = utils.trim(token)
        if not token:
            logger.error(f"[OneAPI] cannot get account due to token is empty, domain: {self.domain}")
            return None

        headers = self._get_headers(key_name=key_name, token=token, token_type=token_type)
        user = self.__get_user_info(headers=headers)
        if not user:
            logger.error(f"[OneAPI] cannot get account with token, domain: {self.domain}")
            return None

        username = user.get("username", "")
        email = user.get("email", "")
        token = user.get("access_token", "") or token

        quota = user.get("quota", 0)
        used_quota = user.get("used_quota", 0)
        available = quota > used_quota

        return Account(
            domain=self.domain,
            username=username,
            password="",
            email=email,
            token=token,
            available=available,
            quota=(quota - used_quota) / 500000,
        )

    def _login(self, username: str, password: str) -> Account:
        username = utils.trim(username)
        password = utils.trim(password)
        if not username or not password:
            logger.error(f"[OneAPI] login failed, username or password is empty, domain: {self.domain}")
            return None

        url = f"{self.domain}/api/user/login"
        payload = {"username": username, "password": password}

        response = utils.http_post_noerror(url=url, headers=self.headers, params=payload, timeout=15)
        data = utils.read_response(response=response, expected=200, deserialize=True)
        if not data or not isinstance(data, dict) or not data.get("success", False):
            logger.error(f"[OneAPI] login failed, domain: {self.domain}, username: {username}")
            return None

        cookie = response.getheader("Set-Cookie")
        token = data.get("access_token", "") or self.__get_access_token(headers=self._get_headers(cookie=cookie))
        if token:
            # quota is 0 when login, so we need to get user info to update quota
            account = self._get_account_by_token(token=token)
            account.password = account.password or password
            return account

        quota = data.get("quota", 0)
        used_quota = data.get("used_quota", 0)
        available = quota > used_quota

        email = data.get("email", "")

        return Account(
            domain=self.domain,
            username=username,
            password=password,
            email=email,
            token=token,
            cookie=cookie,
            available=available,
            quota=(quota - used_quota) / 500000,
        )

    def __query_exist_apikeys(self, headers: dict, page: int = 0, size: int = 10) -> list[str]:
        page, size = max(0, page), max(1, size)
        url = f"{self.domain}/api/token/?p=0&size=10"
        content = utils.http_get(url=f"{url}?p={page}&size={size}", headers=headers, interval=1)

        try:
            data = json.loads(content).get("data", [])
            if data and isinstance(data, list):
                return [f'sk-{item.get("key")}' for item in data if isinstance(item, dict) and "key" in item]
        except:
            logger.error(f"[OneAPI] query api keys failed, domain: {self.domain}")

        return []

    def __get_user_info(self, headers: dict) -> dict:
        url = f"{self.domain}/api/user/self"
        content = utils.http_get(url=url, headers=headers, interval=1)

        try:
            response = json.loads(content)
            data = response.get("data", {})
            return data if data and isinstance(data, dict) else {}
        except:
            logger.error(f"[OneAPI] get user info failed, domain: {self.domain}")
            return {}

    def __get_access_token(self, headers: dict) -> str:
        # query existing access token
        user = self.__get_user_info(headers=headers)
        token = utils.trim(user.get("access_token", "")) if user else ""

        if not token:
            url = f"{self.domain}/api/user/token"
            content = utils.http_get(url=url, headers=headers)

            try:
                data = json.loads(content) if content else {}
                if data and data.get("success", True):
                    token = utils.trim(data.get("data", ""))
            except:
                logger.error(f"[OneAPI] generate access token failed, domain: {self.domain}")

        return token

    def _get_api_keys(self, token: str, token_type: str = "Bearer", cookie: str = "") -> list[str]:
        token, cookie = utils.trim(token), utils.trim(cookie)
        if not token and not cookie:
            logger.error(f"[OneAPI] cannot create api key due to token and cookie is empty, domain: {self.domain}")
            return []

        headers = self._get_headers(token=token, token_type=token_type, cookie=cookie)

        # query existing api keys
        apikeys = self.__query_exist_apikeys(headers=headers)
        if not apikeys:
            url = f"{self.domain}/api/token/"
            name = utils.random_chars(8, punctuation=False)
            payload = {"name": name, "remain_quota": 500000, "expired_time": -1, "unlimited_quota": True}

            # generate new api key
            response = utils.http_post_noerror(url=url, headers=headers, params=payload, timeout=15)
            success = utils.read_response(response=response, expected=200, deserialize=True, key="success")
            if success:
                apikeys = self.__query_exist_apikeys(headers=headers)
            else:
                logger.error(f"[OneAPI] failed to create api key, domain: {self.domain}")

        return [] if not apikeys else apikeys

    def _key_as_token(self) -> bool:
        return True

    def _get_account(
        self, token: str = "", username: str = "", password: str = "", email: str = "", **kwargs
    ) -> Account:
        account = super()._get_account(token, username, password, email, **kwargs)
        if account and not account.available and account.username != "root":
            logger.warning(f"[OneAPI] account {account.username} cannot be used, domain: {self.domain}")
            account = None

        return account

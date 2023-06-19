# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2023-06-17

import base64
import hashlib
import os
import re
from os import getenv
from urllib.parse import parse_qs, urlparse

from certifi import where
import requests

import utils
from logger import logger

OPENAI_API_PRIFIX = getenv("OPENAI_API_PREFIX", "https://api.openai.com")


# reference: https://zhile.io/2023/05/19/how-to-get-chatgpt-access-token-via-pkce.html
# implement: https://github.com/pengzhile/pandora/blob/master/src/pandora/openai/auth.py
def generate_code_verifier() -> str:
    # 随机生成一个长度为 32 的 code_verifier
    token = os.urandom(32)
    code_verifier = base64.urlsafe_b64encode(token).rstrip(b"=")
    return code_verifier.decode("utf-8")


def generate_code_challenge(code_verifier: str) -> str:
    # 对 code_verifier 进行哈希处理，然后再进行 base64url 编码，生成 code_challenge
    m = hashlib.sha256()
    m.update(code_verifier.encode("utf-8"))
    code_challenge = base64.urlsafe_b64encode(m.digest()).rstrip(b"=")
    return code_challenge.decode("utf-8")


def query_sessionid(email: str, password: str) -> str:
    try:
        access_token = OpenAIAuth(email=email, password=password).auth(login_local=True)
        if utils.isblank(access_token):
            logger.error(
                f"[OpenAI] cannot get session due to access token is empty, email: {email}"
            )
            return ""

        url = "{}/dashboard/onboarding/login".format(OPENAI_API_PRIFIX)
        headers = {
            "Accept": "*/*",
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Origin": "https://platform.openai.com",
            "Referer": "https://platform.openai.com/",
            "User-Agent": utils.USER_AGENT,
        }

        resp = requests.post(
            url=url, headers=headers, json={}, allow_redirects=False, timeout=10
        )
        status = resp.status_code if resp else 400
        if status == 200:
            session = resp.json().get("user", {}).get("session", {})
            return session.get("sensitive_id", "")
        else:
            logger.error(
                f"[OpenAI] invalid session response for email: {email}, status: {status}"
            )
            return ""
    except Exception as e:
        logger.error(e)


def create_apikey_once(email: str, password: str) -> tuple[str, bool]:
    session = query_sessionid(email=email, password=password)
    if utils.isblank(session):
        logger.error(
            f"[OpenAI] cannot create new secret key because of session is empty for email: {email}"
        )
        return "", False

    try:
        url = "{}/dashboard/billing/credit_grants".format(OPENAI_API_PRIFIX)
        headers = {
            "Accept": "*/*",
            "Authorization": f"Bearer {session}",
            "Origin": "https://platform.openai.com",
            "Referer": "https://platform.openai.com/",
            "User-Agent": utils.USER_AGENT,
        }
        resp = requests.get(url=url, headers=headers, allow_redirects=False, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            available = data.get("total_available", 0.0)
            if available <= 0:
                granted = data.get("total_granted", 0.0)
                used = data.get("total_used", 0.0)
                logger.error(
                    f"[OpenAI] account has been expired, email: {email} granted: ${granted} used: ${used}"
                )
                return "", True

        url = "{}/dashboard/user/api_keys".format(OPENAI_API_PRIFIX)
        headers["Content-Type"] = "application/json"
        payload = {
            "action": "create",
            "name": utils.random_chars(length=8, punctuation=False),
        }

        resp = requests.post(
            url=url, headers=headers, json=payload, allow_redirects=False, timeout=10
        )
        if resp.status_code != 200:
            logger.error(
                f"[OpenAI] failed to create new secret key for email: {email}, message: {resp.text}"
            )
            return "", False

        data = resp.json()
        if data.get("result", "") != "success":
            logger.error(
                f"[OpenAI] failed to create new secret key for email: {email}, message: {data}"
            )
            return "", False

        return data.get("key", {}).get("sensitive_id", ""), False
    except:
        logger.error(
            f"[OpenAI] cannot query and create new secret key for email: {email}"
        )
        return "", False


def create_apikey(email: str, password: str, retry: int = 1) -> str:
    if utils.isblank(email) or utils.isblank(password):
        return ""

    apikey, expired, retry = "", False, max(retry, 1)
    while not apikey and not expired and retry > 0:
        apikey, expired = create_apikey_once(email=email, password=password)
        retry -= 1

    return apikey


def check_apikey(apikey: str) -> bool:
    regex = r"^sk-[A-Za-z0-9]{48}$"
    if not re.fullmatch(regex, apikey):
        return False

    url = f"{OPENAI_API_PRIFIX}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {apikey}"}
    payload = {
        "model": "gpt-3.5-turbo",
        "max_tokens": 5,
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a Python developer"},
            {"role": "user", "content": "print('Hello, world!')"},
        ],
    }

    resp = requests.post(url=url, headers=headers, json=payload, allow_redirects=False)
    return resp.status_code not in [401, 429]


class OpenAIAuth:
    def __init__(self, email: str, password: str, proxy: str = None, mfa: str = None):
        self.email = email
        self.password = password
        self.mfa = mfa
        self.session = requests.Session()
        self.req_kwargs = {
            "proxies": {
                "http": proxy,
                "https": proxy,
            }
            if proxy
            else None,
            "verify": where(),
            "timeout": 60,
        }
        self.user_agent = utils.USER_AGENT
        self.api_prefix = OPENAI_API_PRIFIX

    @staticmethod
    def __verify(email: str) -> bool:
        regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
        return re.fullmatch(regex, email) is not None

    def auth(self, login_local: bool = True) -> str:
        if not self.__verify(self.email) or not self.password:
            raise Exception(f"[OpenAI] invalid email: {self.email} or password")

        return self.__do_auth() if login_local else self.get_access_token_proxy()

    def __do_auth(self) -> str:
        code_verifier = getenv("CODE_VERIFIER", generate_code_verifier())
        code_challenge = getenv(
            "CODE_CHALLENGE", generate_code_challenge(code_verifier)
        )

        url = (
            "https://auth0.openai.com/authorize?client_id=pdlLIX2Y72MIl2rhLhTE9VV9bN905kBh&audience=https%3A%2F"
            "%2Fapi.openai.com%2Fv1&redirect_uri=com.openai.chat%3A%2F%2Fauth0.openai.com%2Fios%2Fcom.openai.chat"
            "%2Fcallback&scope=openid%20email%20profile%20offline_access%20model.request%20model.read"
            "%20organization.read%20offline&response_type=code&code_challenge={}"
            "&code_challenge_method=S256&prompt=login".format(code_challenge)
        )
        return self.__get_state(code_verifier, url)

    def __get_state(self, code_verifier: str, url: str) -> str:
        headers = {
            "User-Agent": self.user_agent,
            "Referer": "https://ios.chat.openai.com/",
        }
        resp = self.session.get(
            url, headers=headers, allow_redirects=True, **self.req_kwargs
        )

        if resp.status_code == 200:
            try:
                url_params = parse_qs(urlparse(resp.url).query)
                state = url_params["state"][0]
                return self.__check_email(code_verifier, state)
            except IndexError as exc:
                raise Exception(
                    f"[OpenAI] rate limit hit, email: {self.email}"
                ) from exc
        else:
            raise Exception(
                f"[OpenAI] error request login url for email: {self.email} statue: {resp.status_code}"
            )

    def __check_email(self, code_verifier: str, state: str) -> str:
        url = "https://auth0.openai.com/u/login/identifier?state=" + state
        headers = {
            "User-Agent": self.user_agent,
            "Referer": url,
            "Origin": "https://auth0.openai.com",
        }
        data = {
            "state": state,
            "username": self.email,
            "js-available": "true",
            "webauthn-available": "true",
            "is-brave": "false",
            "webauthn-platform-available": "false",
            "action": "default",
        }
        resp = self.session.post(
            url, headers=headers, data=data, allow_redirects=False, **self.req_kwargs
        )

        if resp.status_code == 302:
            return self.__login(code_verifier, state)
        else:
            raise Exception(f"[OpenAI] error check email: {self.email}")

    def __login(self, code_verifier: str, state: str) -> str:
        url = "https://auth0.openai.com/u/login/password?state=" + state
        headers = {
            "User-Agent": self.user_agent,
            "Referer": url,
            "Origin": "https://auth0.openai.com",
        }
        data = {
            "state": state,
            "username": self.email,
            "password": self.password,
            "action": "default",
        }

        resp = self.session.post(
            url, headers=headers, data=data, allow_redirects=False, **self.req_kwargs
        )
        if resp.status_code == 302:
            location = resp.headers["Location"]
            if not location.startswith("/authorize/resume?"):
                raise Exception(f"[OpenAI] login failed, email: {self.email}")

            return self.__login_callback(code_verifier, location, url)

        if resp.status_code == 400:
            raise Exception(f"[OpenAI] wrong email: {self.email} or password")
        else:
            raise Exception(f"[OpenAI] error login, email: {self.email}")

    def __login_callback(self, code_verifier: str, location: str, ref: str) -> str:
        url = "https://auth0.openai.com" + location
        headers = {
            "User-Agent": self.user_agent,
            "Referer": ref,
        }

        resp = self.session.get(
            url, headers=headers, allow_redirects=False, **self.req_kwargs
        )
        if resp.status_code == 302:
            location = resp.headers["Location"]
            if location.startswith("/u/mfa-otp-challenge?"):
                if not self.mfa:
                    raise Exception(f"[OpenAI] MFA required, email: {self.email}")

                return self.__mfa_auth(code_verifier, location)

            if not location.startswith(
                "com.openai.chat://auth0.openai.com/ios/com.openai.chat/callback?"
            ):
                raise Exception(f"[OpenAI] login callback failed, email: {self.email}")

            return self.get_access_token(code_verifier, resp.headers["Location"])

        raise Exception(f"[OpenAI] error login, email: {self.email}")

    def __mfa_auth(self, code_verifier: str, location: str) -> str:
        url = "https://auth0.openai.com" + location
        data = {
            "state": parse_qs(urlparse(url).query)["state"][0],
            "code": self.mfa,
            "action": "default",
        }
        headers = {
            "User-Agent": self.user_agent,
            "Referer": url,
            "Origin": "https://auth0.openai.com",
        }

        resp = self.session.post(
            url, headers=headers, data=data, allow_redirects=False, **self.req_kwargs
        )
        if resp.status_code == 302:
            location = resp.headers["Location"]
            if not location.startswith("/authorize/resume?"):
                raise Exception(f"[OpenAI] MFA failed, email: {self.email}")

            return self.__login_callback(code_verifier, location, url)

        if resp.status_code == 400:
            raise Exception(f"[OpenAI] wrong MFA code for email: {self.email}")
        else:
            raise Exception(f"[OpenAI] error login with email: {self.email}")

    def get_access_token(self, code_verifier: str, callback_url: str) -> str:
        url_params = parse_qs(urlparse(callback_url).query)

        if "error" in url_params:
            error = url_params["error"][0]
            error_description = (
                url_params["error_description"][0]
                if "error_description" in url_params
                else ""
            )
            raise Exception(
                f"[OpenAI] email: {self.email} {error}: {error_description}"
            )

        if "code" not in url_params:
            raise Exception(
                f"[OpenAI] error get code from callback url for email: {self.email}"
            )

        url = "https://auth0.openai.com/oauth/token"
        headers = {
            "User-Agent": self.user_agent,
        }
        data = {
            "redirect_uri": "com.openai.chat://auth0.openai.com/ios/com.openai.chat/callback",
            "grant_type": "authorization_code",
            "client_id": "pdlLIX2Y72MIl2rhLhTE9VV9bN905kBh",
            "code": url_params["code"][0],
            "code_verifier": code_verifier,
        }
        resp = self.session.post(
            url, headers=headers, json=data, allow_redirects=False, **self.req_kwargs
        )

        if resp.status_code == 200:
            access_token = resp.json().get("access_token", "")
            if not access_token:
                raise Exception(
                    f"[OpenAI] get access token failed for email: {self.email}, maybe you need a proxy"
                )

            return access_token
        else:
            raise Exception(f"[OpenAI] email: {self.email} status: {resp.status_code}")

    def get_access_token_proxy(self) -> str:
        url = "https://ai.fakeopen.com/auth/login"
        headers = {
            "User-Agent": self.user_agent,
        }
        data = {
            "username": self.email,
            "password": self.password,
        }
        resp = self.session.post(
            url=url,
            headers=headers,
            data=data,
            allow_redirects=False,
            **self.req_kwargs,
        )

        if resp.status_code == 200:
            json = resp.json()
            if "accessToken" not in json:
                raise Exception(
                    f"[OpenAI] get access token failed, email: {self.email}"
                )

            return json["accessToken"]
        else:
            raise Exception(f"[OpenAI] error get access token, email: {self.email}")


def batch_create(accounts: dict) -> list[str]:
    if not accounts or type(accounts) != dict:
        return []

    tasks = [[k, v] for k, v in accounts.items() if k and v]
    secretkeys = utils.multi_thread_collect(func=create_apikey, params=tasks)
    return [x for x in secretkeys if x]

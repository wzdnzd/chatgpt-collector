# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-03-27

import json
import os
import re
import traceback
from dataclasses import dataclass
from urllib import request
from urllib.error import HTTPError
from urllib.parse import urljoin

import utils
from logger import logger

NOPECHA_KEY = utils.trim(os.environ.get("NOPECHA_KEY", ""))


@dataclass
class CheckInResult(object):
    # 站点
    url: str

    # 是否成功
    success: bool = False

    # 剩余额度，单位：$
    quota: float = 0.0

    # 使用额度，单位：$
    usage: float = 0.0

    # 账号是否存活
    alive: bool = True

    # token
    token: str = ""


def checkin(params: dict) -> list[str]:
    if type(params) != dict:
        return []

    tasks = []
    for k, v in params.items():
        url = utils.trim(k)
        option = v if v is not None and type(v) == dict else {}
        enable = option.get("enable", True)

        if not enable or not utils.isurl(url=url):
            logger.error(f"[ONEAPI] skip execute checkin because disabled or invalid url: {url}")
            continue

        tokens = option.get("tokens", [])
        if not tokens or type(tokens) != list:
            logger.error(f"[ONEAPI] skip execute checkin url: {url} because cannot found any valid token")
            continue

        path, unit = utils.trim(option.get("path", "")), 500000
        if "unit" in option:
            try:
                unit = float(option.get("unit"))
            except:
                logger.error(f"[ONEAPI] invalid unit value: {option.get('unit')}, use default value 500000")

        sitekey = utils.trim(option.get("sitekey", ""))
        sitelink = utils.trim(option.get("sitelink", ""))
        regex = utils.trim(option.get("done", ""))
        account_required = option.get("account", False)

        for item in tokens:
            token = utils.trim(item)
            if token:
                tasks.append((url, path, token, unit, 3, sitekey, sitelink, regex, account_required))
            else:
                logger.error(f"[ONEAPI] ignore invalid token: {item}, url: {url}")

    if not tasks:
        logger.error(f"[ONEAPI] skip execute checkin because no valid task found")
        return []

    logger.info(f"[ONEAPI] start to execute checkin with {len(tasks)} tasks")
    try:
        results = utils.multi_thread_run(checkin_one, tasks)
        failed = [r for r in results if not r.success]
        output = f"[ONEAPI] all tasks completed, total: {len(tasks)}, success: {len(results) - len(failed)}, fail: {len(failed)}"
        if failed:
            sites = [f"url: {r.url}, token: {mask(r.token)}" for r in failed]
            output += f", sites: {sites}"

        logger.info(output)
    except:
        logger.error(f"[ONEAPI] occur error when execute checkin, message:\n {traceback.format_exc()}")

    return []


def checkin_one(
    base: str,
    path: str,
    token: str,
    unit: float,
    retry: int = 3,
    sitekey: str = "",
    sitelink: str = "",
    done_regex: str = "",
    account_required: bool = False,
) -> CheckInResult:
    if not base or not token:
        logger.error(f"[ONEAPI] skip execute checkin because invalid url or token, url: {base}, token: {token}")
        return CheckInResult(url=base, success=False, token=token)

    headers = {"Authorization": f"Bearer {token}", "User-Agent": utils.USER_AGENT}
    url = urljoin(base, path) if path else base

    # to minimize potential nopecha calls, check ahead to see if you've already checked in
    done_regex = utils.trim(done_regex) or "已经?签到"
    content = utils.http_get(url=url, headers=headers, expeced=202)
    try:
        if content and re.search(done_regex, content, flags=re.M) is not None:
            logger.info(f"[ONEAPI] task was ignored because it had already been checked in, url: {url}")

            return CheckInResult(url=base, success=False, token=token)
    except:
        pass

    # skip cloudflare turnstile
    sitekey = utils.trim(sitekey)
    sitelink = utils.trim(sitelink)

    if sitelink and not (sitelink.startswith("https://") or sitelink.startswith("http://")):
        sitelink = urljoin(base, sitelink)

    turnstile = ""
    if NOPECHA_KEY and sitekey and sitelink:
        try:
            import nopecha

            nopecha.api_key = NOPECHA_KEY

            turnstile = nopecha.Token.solve(type="turnstile", sitekey=sitekey, url=sitelink)
        except:
            logger.error(f"[ONEAPI] skip cloudflare turnstile failed , url: {sitelink}")

    if turnstile:
        url = f"{url}?turnstile={turnstile}"

    payload = None
    if account_required:
        account = account_info(base=base, headers=headers)
        if not account:
            logger.error(f"[ONEAPI] cannot signin because fetch account failed")
            return CheckInResult(url=base, success=False, token=token)

        payload = json.dumps(account).encode(encoding="UTF8")

    # checkin
    response, retry = None, max(1, retry)
    while not response and retry > 0:
        retry -= 1
        try:
            req = request.Request(url=url, data=payload, headers=headers, method="POST")
            response = request.urlopen(req, timeout=10, context=utils.CTX)
        except HTTPError as e:
            if e.code in [400, 401, 404]:
                response = e
                break
        except:
            logger.error(f"[ONEAPI] occur error when execute checkin, site: {base}")

    success, message = False, "unknown error"
    if response:
        status, content = response.getcode(), None
        try:
            content = response.read().decode("UTF8")
            data = json.loads(content)

            success = data.get("success", status == 200)
            message = data.get("message", "") or content
        except:
            success = status == 200
            if content:
                message = content

    alive, quota, usage, flag = True, -1, -1, False

    # query quota
    data = account_info(base=base, headers=headers)
    if data and isinstance(data, dict):
        flag = True

        try:
            unit = 1 if unit <= 0 else unit
            alive = data.get("status", 0) == 1
            quota = round(data.get("quota", 0) / unit, 2)
            usage = round(data.get("used_quota", 0) / unit, 2)
        except:
            logger.error(f"[ONEAPI] cannot parse quota from response: {content}")

    quota_info = f"${quota}" if flag else "unknown"
    usage_info = f"${usage}" if flag else "unknown"

    logger.info(
        f"[ONEAPI] checkin finished, url: {base}, token: {mask(token)}, success: {success}, alive: {alive if flag else 'unknown'}, quota: {quota_info}, usage: {usage_info}, messsage: {message}"
    )

    return CheckInResult(url=base, success=success, alive=alive, quota=quota, usage=usage, token=token)


def account_info(base: str, headers: dict = None, token: str = "") -> dict:
    token = utils.trim(token)

    if not token and not headers:
        return None
    elif not headers:
        headers = {"Authorization": f"Bearer {token}", "User-Agent": utils.USER_AGENT}

    url = urljoin(utils.trim(base), "/api/user/self")
    data, content = None, utils.http_get(url=url, headers=headers)

    if content:
        try:
            result = json.loads(content)
            if not result.get("success", False):
                logger.error(f"[ONEAPI] cannot get account information, message: {result.get('message', '')}")
            else:
                data = result.get("data", {})
        except:
            logger.error(f"[ONEAPI] cannot parse account information from response: {content}")

    return data


def mask(text: str, n: int = 10) -> str:
    content, n = utils.trim(text), max(1, n)
    return content if len(content) <= n else content[:-n] + "*" * n

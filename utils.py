# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2022-07-15

import gzip
import json
import multiprocessing
import os
import random
import re
import ssl
import string
import time
import typing
import urllib
import urllib.parse
import urllib.request
from http.client import HTTPMessage, HTTPResponse
from threading import Lock

from logger import logger
from urlvalidator import isurl

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"

PATH = os.path.abspath(os.path.dirname(__file__))

FILE_LOCK = Lock()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def http_error_302(
        self,
        req: urllib.request.Request,
        fp: typing.IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
    ) -> typing.IO[bytes]:
        return fp


def http_get(
    url: str,
    headers: dict = None,
    params: dict = None,
    retry: int = 3,
    proxy: str = "",
    interval: float = 0,
) -> str:
    if not isurl(url=url):
        logger.error(f"invalid url: {url}")
        return ""

    if retry <= 0:
        return ""

    if not headers:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        }

    interval = max(0, interval)
    try:
        url = encoding_url(url=url)
        if params and isinstance(params, dict):
            data = urllib.parse.urlencode(params)
            if "?" in url:
                url += f"&{data}"
            else:
                url += f"?{data}"

        request = urllib.request.Request(url=url, headers=headers)
        if proxy and (proxy.startswith("https://") or proxy.startswith("http://")):
            host, protocal = "", ""
            if proxy.startswith("https://"):
                host, protocal = proxy[8:], "https"
            else:
                host, protocal = proxy[7:], "http"
            request.set_proxy(host=host, type=protocal)

        response = urllib.request.urlopen(request, timeout=10, context=CTX)
        content = response.read()
        status_code = response.getcode()
        try:
            content = str(content, encoding="utf8")
        except:
            content = gzip.decompress(content).decode("utf8")

        return content if status_code == 200 else ""
    except Exception:
        time.sleep(interval)
        return http_get(
            url=url,
            headers=headers,
            params=params,
            retry=retry - 1,
            proxy=proxy,
            interval=interval,
        )


def http_post(
    url: str,
    headers: dict = None,
    params: dict = {},
    retry: int = 3,
    timeout: float = 6,
    allow_redirects: bool = True,
) -> tuple[HTTPResponse, int]:
    if params is None or type(params) != dict:
        return None, 1

    timeout, retry = max(timeout, 1), retry - 1
    if not headers:
        headers = {
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
        }
    try:
        data = json.dumps(params).encode(encoding="UTF8")
        request = urllib.request.Request(
            url=url, data=data, headers=headers, method="POST"
        )
        if allow_redirects:
            return urllib.request.urlopen(request, timeout=timeout, context=CTX), 0

        opener = urllib.request.build_opener(NoRedirect)
        return opener.open(request, timeout=timeout), 0
    except urllib.error.HTTPError as e:
        if retry < 0 or e.code in [400, 401, 404, 405]:
            return None, 2

        return http_post(
            url=url,
            headers=headers,
            params=params,
            retry=retry,
            allow_redirects=allow_redirects,
        )
    except (TimeoutError, urllib.error.URLError) as e:
        return None, 3
    except Exception:
        if retry < 0:
            return None, 3
        return http_post(
            url=url,
            headers=headers,
            params=params,
            retry=retry,
            allow_redirects=allow_redirects,
        )


def http_post_noerror(
    url: str,
    headers: dict = None,
    params: dict = {},
    retry: int = 3,
    timeout: float = 6,
    allow_redirects: bool = True,
) -> HTTPResponse:
    response, _ = http_post(
        url=url,
        headers=headers,
        params=params,
        retry=retry,
        timeout=timeout,
        allow_redirects=allow_redirects,
    )
    return response


def extract_domain(url: str, include_protocal: bool = True) -> str:
    if not url:
        return ""

    start = url.find("//")
    if start == -1:
        start = -2

    end = url.find("/", start + 2)
    if end == -1:
        end = len(url)

    if include_protocal:
        return url[:end]

    return url[start + 2 : end]


def encoding_url(url: str) -> str:
    if not url:
        return ""

    url = url.strip()

    # 正则匹配中文汉字
    cn_chars = re.findall("[\u4e00-\u9fa5]+", url)
    if not cn_chars:
        return url

    # 遍历进行 punycode 编码
    punycodes = list(
        map(lambda x: "xn--" + x.encode("punycode").decode("utf-8"), cn_chars)
    )

    # 对原 url 进行替换
    for c, pc in zip(cn_chars, punycodes):
        url = url[: url.find(c)] + pc + url[url.find(c) + len(c) :]

    return url


def isblank(text: str) -> bool:
    return not text or type(text) != str or not text.strip()


def trim(text: str) -> str:
    if not text or type(text) != str:
        return ""

    return text.strip()


def load_dotenv() -> None:
    path = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(path, ".env")

    if not os.path.exists(filename) or os.path.isdir(filename):
        return

    with open(filename, mode="r", encoding="utf8") as f:
        for line in f.readlines():
            content = line.strip()
            if not content or content.startswith("#") or "=" not in content:
                continue
            words = content.split("=", maxsplit=1)
            k, v = words[0].strip(), words[1].strip()
            if k and v:
                os.environ[k] = v


def url_complete(site: str) -> str:
    if isblank(site):
        return ""

    if not site.startswith("https://"):
        # force use https protocal
        if site.startswith("http://"):
            site = site.replace("http://", "https://")
        else:
            site = f"https://{site}"

    return extract_domain(url=site, include_protocal=True)


def multi_thread_collect(func: typing.Callable, params: list) -> list:
    if not func or not params or type(params) != list:
        return []

    cpu_count = multiprocessing.cpu_count()
    num = len(params) if len(params) <= cpu_count else cpu_count

    pool, starttime = multiprocessing.Pool(num), time.time()
    if type(params[0]) == list or type(params[0]) == tuple:
        results = pool.starmap(func, params)
    else:
        results = pool.map(func, params)
    pool.close()

    funcname = getattr(func, "__name__", repr(func))
    logger.info(
        f"concurrent execute [{funcname}] finished, count: {len(params)}, cost: {time.time()-starttime:.2f}s"
    )

    return results


def random_chars(length: int, punctuation: bool = False) -> str:
    length = max(length, 1)
    if punctuation:
        chars = "".join(
            random.sample(
                string.ascii_letters + string.digits + string.punctuation, length
            )
        )
    else:
        chars = "".join(random.sample(string.ascii_letters + string.digits, length))

    return chars


def write_file(filename: str, lines: str | list, overwrite: bool = True) -> bool:
    if not filename or not lines or type(lines) not in [str, list]:
        logger.error(f"filename or lines is invalid, filename: {filename}")
        return False

    try:
        if not isinstance(lines, str):
            lines = "\n".join(lines)

        filepath = os.path.abspath(os.path.dirname(filename))
        os.makedirs(filepath, exist_ok=True)
        mode = "w" if overwrite else "a"

        # waitting for lock
        FILE_LOCK.acquire(30)

        with open(filename, mode, encoding="UTF8") as f:
            f.write(lines + "\n")
            f.flush()

        # release lock
        FILE_LOCK.release()

        return True
    except:
        return False


def is_number(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False

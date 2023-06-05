import re
import json

import utils
from aes import AESCipher
from logger import logger


def collect(params: dict) -> list[str]:
    params = {} if type(params) != dict else params
    url = utils.trim(
        params.get(
            "url", "https://chatgpt-site.zhaoyeqing.cn/static/js/main.39192abd.js"
        )
    )

    salt = utils.trim(params.get("passphrase", ""))
    prefix = "https://api.zhaoyeqing.cn"

    content = utils.http_get(url=url)
    if not content:
        logger.warn(f"[ZYQ] failed to fetch content from url: {url}")
    else:
        regex = r"AES\.decrypt\(e\.web_url,\"(.*?)\"\)\.toString"
        match = re.findall(regex, content, flags=re.I)
        passphrase = match[0] if match else salt

        pattern = r"baseURL:\"(https://.*?)\""
        group = re.findall(pattern, content, flags=re.I)
        domain = utils.trim(group[0]) if group else ""
        prefix = domain if domain else prefix

    if utils.isblank(passphrase):
        logger.error(f"[ZYQ] cannot decrypt due to passphrase is empty")
        return []

    # https://api.zhaoyeqing.cn/site/list/chatgpt
    url = f"{prefix}/site/list/chatgpt"

    # fetch json data
    content, sites = utils.http_get(url=url), set()
    if not content:
        logger.error(f"[ZYQ] fetch websites json data error, url: {url}")
        return []

    types = params.get("types", [])
    include = utils.trim(params.get("include", ""))
    exclude = utils.trim(params.get("exclude", ""))
    sitetypes = set(types) if types else None
    try:
        data = json.loads(content)
        cipher = AESCipher(passphrase=passphrase)
        for item in data:
            if not item or (sitetypes and item.get("key", "") not in sitetypes):
                continue

            website = item.get("web_url", "")
            try:
                url = utils.url_complete(site=cipher.decrypt(website))
                if not url:
                    continue

                if include and not re.search(include, url, re.I):
                    continue
                else:
                    if exclude and re.search(exclude, url, re.I):
                        continue

                sites.add(url)
            except:
                logger.warn(
                    f"[ZYQ] invalid web url: {website} or include: {include} or exclude: {exclude} is invalid"
                )
    except:
        logger.error(f"[ZYQ] occue error when parse data")

    logger.info(f"[ZYQ] crawl finished, found {len(sites)} sites")
    return list(sites)

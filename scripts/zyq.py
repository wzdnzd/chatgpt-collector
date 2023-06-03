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

    content = utils.http_get(url=url)
    if not content:
        logger.error(f"[ZYQ] failed to fetch content from url: {url}")
        return []

    salt = utils.trim(params.get("passphrase", ""))
    regex = r"AES\.decrypt\(e\.web_url,\"(.*?)\"\)\.toString"
    match = re.findall(regex, content, flags=re.I)
    passphrase = match[0] if match else salt
    if utils.isblank(passphrase):
        logger.error(f"[ZYQ] cannot decrypt due to passphrase is empty")
        return []

    prefix, regex = "https://api.zhaoyeqing.cn", r"baseURL:\"(https://.*?)\""
    group = re.findall(regex, content, flags=re.I)
    prefix = utils.trim(group[0]) if group else prefix
    url = f"{prefix}/site/list/chatgpt"

    # fetch json data
    content, sites = utils.http_get(url=url), set()
    if not content:
        logger.error(f"[ZYQ] fetch websites json data error, url: {url}")
        return []

    types = params.get("types", [])
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
                if url:
                    sites.add(url)
            except:
                logger.warn(f"[ZYQ] invalid web url: {website}")
    except:
        logger.error(f"[ZYQ] occue error when parse data")

    logger.info(f"[ZYQ] crawl finished, found {len(sites)} sites")
    return list(sites)

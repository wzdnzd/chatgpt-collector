# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-09-27

import argparse
import os
import time

import requests

FOFA_API = os.environ.get("CUSTOMIZED_FOFA_API", "").strip()


def trim(text: str) -> str:
    if not text or type(text) != str:
        return ""

    return text.strip()


def save_file(filename: str, lines: list) -> bool:
    if not filename or not lines:
        print(f"filename or lines is empty, filename: {filename}")
        return False

    try:
        if not isinstance(lines, str):
            lines = "\n".join(lines)

        filepath = os.path.abspath(os.path.dirname(filename))
        os.makedirs(filepath, exist_ok=True)
        with open(filename, "w+", encoding="UTF8") as f:
            f.write(lines)
            f.flush()

        return True
    except:
        return False


def complete(url: str, secret: bool = False) -> str:
    url = trim(url)
    if not url:
        return ""

    if not url.startswith("https://"):
        if url.startswith("http://"):
            if secret:
                url = url.replace("http://", "https://")
        else:
            url = f"https://{url}"

    return url


def search(keyword: str, filename: str) -> None:
    url = trim(FOFA_API)
    if not url or not (url.startswith("https://") or url.startswith("http://")):
        print(f"invalid fofa API address: {FOFA_API}")
        return

    keyword, filename = trim(keyword), trim(filename)
    if not keyword or not filename:
        print(f"skip search due to invalid keyword: {keyword} or filename: {filename}")
        return

    items, retry = None, 0
    headers = {
        "Accept": "*/*",
        "Content-Type": "text/plain",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    }

    while items is None and retry < 3:
        retry += 1
        response = requests.post(url=url, data=keyword, headers=headers, timeout=15)

        if response.status_code != 200:
            interval = 2**retry
            print(f"status: {response.status_code}, will retry when sleep {interval}s later")

            time.sleep(interval)
            continue

        result = response.json()
        if result.get("error", False):
            print(f"query error, message: {result}")
            continue

        items = result.get("results", [])

    if not items or not isinstance(items, list):
        print(f"cannot search any result, keyword: {keyword}")
        return

    lines = set()
    for item in items:
        if not item or not isinstance(item, list):
            continue

        link = complete(url=item[0])
        if link:
            lines.add(link)

    save_file(filename=filename, lines=list(lines))
    print(f"search finished, found {len(lines)} results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="result filename to be saved",
    )

    parser.add_argument(
        "-k",
        "--keyword",
        type=str,
        required=True,
        help="keywords for search",
    )

    args = parser.parse_args()
    search(keyword=args.keyword, filename=args.file)

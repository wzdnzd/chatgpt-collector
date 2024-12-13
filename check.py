# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-03-26

import argparse
import asyncio
import errno
import json
import os
import time
import traceback
from collections import defaultdict
from random import choice
from typing import Iterable

from tqdm import tqdm

import interactive
import utils
from logger import logger
from provider import SUPPORTED_PROVIDERS
from provider.base import APIStyle, ServiceInfo


def read_in_chunks(filepath: str, chunk_size: int = 100):
    precheck(filepath=filepath)

    chunk_size = max(1, chunk_size)
    with open(filepath, "r", encoding="utf8") as f:
        while True:
            lines = []
            try:
                for _ in range(chunk_size):
                    text = next(f).replace("\n", "")
                    lines.append(text)
            except StopIteration:
                if not lines:
                    break

            yield lines
            if len(lines) < chunk_size:
                break


def count_lines(filepath: str) -> int:
    precheck(filepath=filepath)

    with open(filepath, "r", encoding="utf8") as f:
        return sum(1 for _ in f)


def dedup(filepath: str) -> None:
    def include_subpath(url: str) -> bool:
        url = utils.trim(url).lower()
        if url.startswith("http://"):
            url = url[7:]
        elif url.startswith("https://"):
            url = url[8:]

        return "/" in url and not url.endswith("/")

    def cmp(url: str) -> str:
        x = 1 if include_subpath(url=url) else 0
        y = 2 if url.startswith("https://") else 1 if url.startswith("http://") else 0
        return (x, y, url)

    precheck(filepath=filepath)

    lines, groups, links = [], defaultdict(set), []
    with open(filepath, "r", encoding="utf8") as f:
        lines = f.readlines()

    # filetr and group by domain
    for line in lines:
        line = utils.trim(line).lower()
        if not line or line.startswith("#") or line.startswith(";"):
            continue

        domain = utils.extract_domain(url=line, include_protocal=False)
        if domain:
            if not line.startswith("http://") and not line.startswith("https://"):
                line = "http://" + line

            groups[domain].add(line)

    # under the same domain name, give priority to URLs starting with https://
    for v in groups.values():
        if not v:
            continue

        urls = list(v)
        if len(urls) > 1:
            urls.sort(key=cmp, reverse=True)

        links.append(urls[0])

    total, remain = len(lines), len(links)
    logger.info(f"[Check] finished dedup for file: {filepath}, total: {total}, remain: {remain}, drop: {total-remain}")

    utils.write_file(filename=filepath, lines=links, overwrite=True)


def precheck(filepath: str) -> None:
    filepath = utils.trim(filepath)

    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)


def preprocess(
    source: str,
    provider: str,
    threshold: float = 0,
    num_threads: int = 0,
    show_progress: bool = True,
) -> tuple[str, int]:
    def _preprocess_one(
        domain: str, token: str = "", username: str = "", password: str = "", email: str = "", **kwargs
    ) -> ServiceInfo:
        try:
            obj = SUPPORTED_PROVIDERS[provider](domain=domain)
            return obj.get_service(
                token=token,
                username=username,
                password=password,
                email=email,
                threshold=threshold,
                **kwargs,
            )
        except:
            logger.error(f"[Check] failed to fetch service for domain: {domain}")
            return None

    def _filter_and_concat(candidates: list[ServiceInfo]) -> set[str]:
        result = set()
        if candidates and isinstance(candidates, Iterable):
            for candidate in candidates:
                if (
                    not candidate
                    or not isinstance(candidate, ServiceInfo)
                    or not candidate.available
                    or not candidate.api_urls
                    or not APIStyle.is_standard(candidate.style.name)
                ):
                    continue

                for url in candidate.api_urls:
                    if candidate.api_keys:
                        url += f"?token={choice(candidate.api_keys)}"

                    result.add(url)

        return result

    provider = utils.trim(provider).lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"[Check] unsupported service provider: {provider}")

    threshold = max(0, threshold)
    source = utils.trim(source)

    # dedup source file
    dedup(filepath=source)

    if not os.path.exists(source) or not os.path.isfile(source):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), source)

    olds, news = [], []
    filepath = SUPPORTED_PROVIDERS[provider]._get_default_persist_file()

    if os.path.exists(filepath) and os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf8") as f:
            for line in f.readlines():
                line = utils.trim(line).replace("\n", "")
                if not line or not (line.startswith("{") and line.endswith("}")):
                    continue

                try:
                    service = ServiceInfo.deserialize(content=line)
                    olds.append([service.domain, service.token, service.username, service.password, service.email])
                except:
                    logger.error(f"[Check] failed to deserialize service info: {line}")

    # backup existing file
    utils.backup_file(filepath=filepath, with_time=True)

    services = utils.multi_thread_run(
        func=_preprocess_one,
        tasks=olds,
        num_threads=num_threads,
        show_progress=show_progress,
    )

    records = {} if not services else {x.domain: x for x in services if x}
    with open(source, "r", encoding="utf8") as f:
        for line in f.readlines():
            line = utils.trim(line).replace("\n", "").lower()
            if not line or line.startswith("#") or line.startswith(";") or line in records:
                continue

            if not line.startswith("https://") and not line.startswith("http://"):
                line = "http://" + line

            news.append([line, "", "", "", ""])

    growths = utils.multi_thread_run(
        func=_preprocess_one,
        tasks=news,
        num_threads=num_threads,
        show_progress=show_progress,
    )

    records.update({item.domain: item for item in growths if item})
    passes = [x.to_dict() for x in records.values() if x]
    style = SUPPORTED_PROVIDERS[provider].api_style().value

    # save service info to file
    summary = f"{os.path.splitext(filepath)[0]}-summary.json"
    with open(summary, "w+", encoding="utf8") as f:
        f.write(json.dumps(passes))
        f.flush()

        logger.info(f"[Check] saved service information to file: {summary}")

    final_urls = _filter_and_concat(records.values())
    final_urls.update(_filter_and_concat(growths))
    if not final_urls:
        logger.error(f"[Check] no available service api found for provider: {provider}")
        return "", style

    # generate new source file
    words = os.path.splitext(source)
    filename, extension = words[0], words[1]
    current = time.strftime("%Y%m%d%H%M%S", time.localtime())
    newfile = f"{filename}-{current}{extension}"

    utils.write_file(filename=newfile, lines=list(final_urls), overwrite=True)
    return newfile, style


def main(args: argparse.Namespace) -> None:
    target = utils.trim(args.filename)
    if not target:
        logger.error(f"[Check] cannot fetch candidates due to not specify source file")
        return

    source = os.path.abspath(target)
    model = utils.trim(args.model).lower() or "gpt-3.5-turbo"
    current = time.strftime("%Y%m%d%H%M%S", time.localtime())

    result = utils.trim(args.result)
    if not result:
        result = f"availables-{model}-{current}.txt"

    num_processes, num_threads = args.num, args.thread
    dest = os.path.abspath(os.path.join(os.path.dirname(source), result))

    # generate service api from provider
    changed, style = False, args.style
    if args.provider:
        source, style = preprocess(
            source=source,
            provider=args.provider,
            threshold=args.allocation,
            num_threads=num_threads,
            show_progress=args.display,
        )
        changed = True

        if not source:
            logger.error(f"[Check] failed to generate service api from provider: {args.provider}")
            return

    # merge dest file content into source file if exist
    if os.path.exists(dest) and os.path.isfile(dest):
        with open(dest, "r", encoding="utf8") as f:
            lines = [x.strip().lower().replace("\n", "") for x in f.readlines() if x]
            utils.write_file(filename=source, lines=lines, overwrite=False)

        if args.overwrite:
            utils.backup_file(filepath=dest)

    # dedup candidates
    dedup(filepath=source)

    potentials = utils.trim(args.latent).lower()

    question = utils.trim(args.question)
    keyword = utils.trim(args.keyword)
    if (question and not keyword) or (not question and keyword):
        logger.error(f"[Check] question and keyword must be set together")
        return

    strict = not args.easing
    try:
        if not args.blocked:
            with open(source, mode="r", encoding="utf8") as f:
                sites = [x.replace("\n", "") for x in f.readlines() if x]
                asyncio.run(
                    interactive.check_async(
                        sites=sites,
                        filename=dest,
                        potentials=potentials,
                        wander=args.wander,
                        question=question,
                        keyword=keyword,
                        model=model,
                        concurrency=num_threads,
                        show_progress=args.display,
                        style=style,
                        headers=args.zany,
                        strict=strict,
                    )
                )

        else:
            size, total = max(args.chunk, 1), count_lines(source)
            count = (total + size - 1) // size

            logger.info(f"[Check] start to check available for sites")
            chunks = read_in_chunks(source, size)
            if num_processes == 1 or count == 1:
                tasks = chunks if count == 1 else tqdm(chunks, total=count, desc="Progress", leave=True)
                show = True if count == 1 else False

                for task in tasks:
                    interactive.check_concurrent(
                        sites=task,
                        filename=dest,
                        potentials=potentials,
                        wander=args.wander,
                        question=question,
                        keyword=keyword,
                        model=model,
                        num_threads=num_threads,
                        show_progress=show,
                        index=0,
                        style=style,
                        headers=args.zany,
                        strict=strict,
                    )
            else:
                tasks = [
                    [x, dest, potentials, args.wander, model, num_threads, args.display, i + 1, style, args.zany]
                    for i, x in enumerate(chunks)
                ]
                utils.multi_process_run(func=interactive.check_concurrent, tasks=tasks)

        # dedup result file
        dedup(filepath=dest)

        # remove temporary source file generated by preprocess
        if changed:
            os.remove(source)

        logger.info(f"[Check] check finished, avaiable links will be saved to file {dest} if exists")
    except FileNotFoundError as e:
        logger.error(f"[Check] file {e.filename} not exists")
    except:
        logger.error(f"[Check] batch check error, message:\n{traceback.format_exc()}")


if __name__ == "__main__":
    utils.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--allocation",
        type=float,
        required=False,
        default=0.0,
        help="threshold of the service quota, defaults to 0.0, unit: $",
    )

    parser.add_argument(
        "-b",
        "--blocked",
        dest="blocked",
        action="store_true",
        default=False,
        help="synchronised testing",
    )

    parser.add_argument(
        "-c",
        "--chunk",
        type=int,
        required=False,
        default=512,
        help="chunk size of each file block",
    )

    parser.add_argument(
        "-d",
        "--display",
        dest="display",
        action="store_true",
        default=False,
        help="show check progress bar",
    )

    parser.add_argument(
        "-e",
        "--easing",
        dest="easing",
        action="store_true",
        default=False,
        help="whether to use easing mode",
    )

    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=True,
        default="",
        help="name of the file to be checked",
    )

    parser.add_argument(
        "-k",
        "--keyword",
        type=str,
        required=False,
        default="",
        help="keyword for check answer accuracy, must be set if question is set",
    )

    parser.add_argument(
        "-l",
        "--latent",
        type=str,
        required=False,
        default="",
        help="potential APIs to be tested, multiple APIs separated by commas",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        default="gpt-3.5-turbo",
        help="model name to chat with",
    )

    parser.add_argument(
        "-n",
        "--num",
        type=int,
        required=False,
        default=0,
        help="number of processes, CPU cores as default",
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="overwrite file if exists",
    )

    parser.add_argument(
        "-p",
        "--provider",
        type=str,
        required=False,
        choices=SUPPORTED_PROVIDERS.keys(),
        help="service provider to be checked",
    )

    parser.add_argument(
        "-q",
        "--question",
        type=str,
        required=False,
        default="",
        help="question to ask, must be set if keyword is set",
    )

    parser.add_argument(
        "-r",
        "--result",
        type=str,
        required=False,
        default="",
        help="filename to save results",
    )

    parser.add_argument(
        "-s",
        "--style",
        type=int,
        required=False,
        default=0,
        choices=[0, 1],
        help="request body format, 0 means to use the OpenAI style and 1 means to use the Azure style",
    )

    parser.add_argument(
        "-t",
        "--thread",
        type=int,
        required=False,
        default=0,
        help="number of concurrent threads, defaults to double the number of CPU cores",
    )

    parser.add_argument(
        "-w",
        "--wander",
        dest="wander",
        action="store_true",
        default=False,
        help="whether to use common APIs for probing",
    )

    parser.add_argument(
        "-z",
        "--zany",
        type=str,
        required=False,
        default="",
        help="custom request headers are separated by '|', and the key-value pairs of the headers are separated by ':'",
    )

    main(parser.parse_args())

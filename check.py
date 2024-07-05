# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-03-26

import argparse
import asyncio
import os
import traceback
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

import interactive
import utils
from logger import logger


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
        raise FileNotFoundError(f"[Check] {filepath} was not found or is a directory")


def main(args: argparse.Namespace) -> None:
    target = utils.trim(args.filename)
    if not target:
        logger.error(f"[Check] cannot fetch candidates due to not specify source file")
        return

    source = os.path.abspath(target)
    model = utils.trim(args.model).lower() or "gpt-3.5-turbo"
    current = datetime.now().strftime("%Y%m%d%H%M")

    result = utils.trim(args.result)
    if not result:
        result = f"availables-{model}-{current}.txt"

    dest = os.path.abspath(os.path.join(os.path.dirname(source), result))

    # merge dest file content into source file if exist
    if os.path.exists(dest) and os.path.isfile(dest):
        if not args.newapi:
            with open(dest, "r", encoding="utf8") as f:
                lines = [x.strip().lower().replace("\n", "") for x in f.readlines() if x]
                utils.write_file(filename=source, lines=lines, overwrite=False)

        if args.overwrite:
            backup_file = f"{dest}.bak"
            if os.path.exists(backup_file) and os.path.isfile(backup_file):
                os.remove(backup_file)

            os.rename(dest, backup_file)

    # dedup candidates
    dedup(filepath=source)

    potentials = utils.trim(args.latent).lower()
    num_processes, num_threads = args.process, args.thread

    try:
        if not args.blocked or args.newapi:
            with open(source, mode="r", encoding="utf8") as f:
                sites = [x.replace("\n", "") for x in f.readlines() if x]
                if args.newapi:
                    items = [[x, dest] for x in sites]
                    utils.multi_thread_run(
                        func=interactive.burst_newapi,
                        tasks=items,
                        num_threads=num_threads,
                        show_progress=args.display,
                    )
                else:
                    asyncio.run(
                        interactive.check_async(
                            sites=sites,
                            filename=dest,
                            potentials=potentials,
                            wander=args.wander,
                            model=model,
                            concurrency=num_threads,
                            show_progress=args.display,
                            style=args.style,
                            headers=args.zany,
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
                        model=model,
                        num_threads=num_threads,
                        show_progress=show,
                        index=0,
                        style=args.style,
                        headers=args.zany,
                    )
            else:
                tasks = [
                    [x, dest, potentials, args.wander, model, num_threads, args.display, i + 1, args.style, args.zany]
                    for i, x in enumerate(chunks)
                ]
                utils.multi_process_run(func=interactive.check_concurrent, tasks=tasks)

        # dedup result file
        dedup(filepath=dest)

        logger.info(f"[Check] check finished, avaiable links will be saved to file {dest} if exists")
    except FileNotFoundError:
        logger.error(f"[Check] file {source} not exists")
    except:
        logger.error(f"[Check] batch check error, message:\n{traceback.format_exc()}")


if __name__ == "__main__":
    utils.load_dotenv()

    parser = argparse.ArgumentParser()
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
        "-n",
        "--newapi",
        dest="newapi",
        action="store_true",
        default=False,
        help="default password detection in NewAPI or OneAPI",
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
        "-o",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="overwrite file if exists",
    )

    parser.add_argument(
        "-p",
        "--process",
        type=int,
        required=False,
        default=0,
        help="number of processes, CPU cores as default",
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

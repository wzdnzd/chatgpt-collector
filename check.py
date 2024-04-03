import argparse
import os
import traceback
from datetime import datetime

from tqdm import tqdm

import utils
from logger import logger
from scripts import nextweb


def available_check(
    lines: list[str],
    saved_file,
    model: str = "gpt-3.5-turbo",
    num_threads: int = -1,
    show_progress: bool = False,
    index: int = 0,
) -> None:
    if not lines or not isinstance(lines, list):
        logger.warning(f"[Check] skip process due to lines is empty")
        return

    saved_file = utils.trim(saved_file)
    if not saved_file:
        raise ValueError(f"you must specify the save file path")

    filepath = os.path.abspath(saved_file)
    sites = nextweb.concurrent_check(
        sites=lines,
        model=model,
        num_threads=num_threads,
        show_progress=show_progress,
        index=index,
    )

    if not sites:
        return

    if not utils.write_file(filepath, sites, overwrite=False):
        logger.error(f"[Check] save result to file {filepath} failed, sites: {sites}")


def read_in_chunks(filepath: str, chunk_size: int = 100):
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        raise FileNotFoundError(f"{filepath} was not found or is a directory")

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
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        raise FileNotFoundError(f"{filepath} was not found or is a directory")

    with open(filepath, "r", encoding="utf8") as f:
        return sum(1 for _ in f)


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

    dest = os.path.abspath(os.path.join(utils.PATH, "data", result))

    if args.overwrite and os.path.exists(dest) and os.path.isfile(dest):
        os.remove(dest)

    try:
        size, total = max(args.chunk, 1), count_lines(source)
        count = (total + size - 1) // size
        num_processes, num_threads = args.process, args.thread

        logger.info(f"[Check] start to check available for sites")
        chunks = read_in_chunks(source, size)
        if num_processes == 1 or count == 1:
            tasks = chunks if count == 1 else tqdm(chunks, total=count, desc="Progress", leave=True)
            show = True if count == 1 else False

            for task in tasks:
                available_check(lines=task, saved_file=dest, model=model, num_threads=num_threads, show_progress=show)
        else:
            tasks = [[x, dest, model, num_threads, args.show, i + 1] for i, x in enumerate(chunks)]
            utils.multi_process_collect(func=available_check, tasks=tasks)

        logger.info(f"[Check] check finished, avaiable links will be saved to file {dest} if exists")
    except FileNotFoundError:
        logger.error(f"[Check] file {source} not exists")
    except:
        logger.error(f"[Check] batch check error, message:\n{traceback.format_exc()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--chunk",
        type=int,
        required=False,
        default=512,
        help="chunk size of each file block",
    )

    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=False,
        default=nextweb.MATERIAL_FILE,
        help="name of the file to be checked",
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
        "--show",
        dest="show",
        action="store_true",
        default=False,
        help="show check progress bar",
    )

    parser.add_argument(
        "-t",
        "--thread",
        type=int,
        required=False,
        default=0,
        help="number of concurrent threads, defaults to double the number of CPU cores",
    )

    main(parser.parse_args())

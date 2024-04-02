import argparse
import os
import traceback
from datetime import datetime

from tqdm import tqdm

import utils
from logger import logger
from scripts import nextweb


def available_check(lines: str, saved_file, model: str = "gpt-3.5-turbo") -> None:
    if not lines or not isinstance(lines, list):
        logger.warning(f"[Check] skip process due to lines is empty")
        return

    saved_file = utils.trim(saved_file)
    if not saved_file:
        raise ValueError(f"you must specify the save file path")

    filepath = os.path.abspath(saved_file)
    model = utils.trim(model) or "gpt-3.5-turbo"

    tasks = [[x, model] for x in lines if x]
    result = utils.multi_thread_collect(func=nextweb.check, tasks=tasks, num_threads=64)
    sites = [x for x in result if x]
    if sites:
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

    filename = os.path.join(utils.PATH, "data", f"availables-{model}-{current}.txt")
    if args.overwrite and os.path.exists(filename) and os.path.isfile(filename):
        os.remove(filename)

    try:
        chunk_size, total = 512, count_lines(source)
        count = (total + chunk_size - 1) // chunk_size

        for chunk in tqdm(read_in_chunks(source, chunk_size), total=count, desc="Progress", leave=True):
            available_check(lines=chunk, saved_file=filename, model=model)
    except FileNotFoundError:
        logger.error(f"[Check] file {source} not exists")
    except:
        logger.error(f"[Check] batch check error, message:\n{traceback.format_exc()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=False,
        default=nextweb.MATERIAL_FILE,
        help="file name of candidates",
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

    main(parser.parse_args())

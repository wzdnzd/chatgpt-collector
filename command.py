# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2022-09-23

import argparse
import json
import os
from datetime import datetime

import push
import utils
from logger import logger
from scripts import nextweb


def main(args: argparse.Namespace) -> None:
    username = utils.trim(args.username)
    if not username:
        logger.error(f"[CMD] username cannot not be empty")
        return

    repository = utils.trim(args.repository)
    if not repository:
        logger.error(f"[CMD] repository cannot not be empty")
        return

    params = {
        "sort": args.sort,
        "refresh": args.generate,
        "checkonly": args.checkonly,
        "overlay": args.overlay,
        "model": args.model,
        "num_threads": args.num,
        "skip_check": args.nocheck,
        "chunk": max(1, args.chunk),
        "async": args.run_async,
        "username": username,
        "repository": repository,
        "standard": args.typical,
        "exclude": utils.trim(args.exclude),
    }

    filename = utils.trim(args.persist)
    if filename:
        filepath = os.path.abspath(filename)
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            logger.error(f"[CMD] persist config file '{filepath}' not exists")
            return

        try:
            with open(filepath, "r", encoding="utf8") as f:
                config = json.load(f)
                server, storages = config.get("server", ""), config.get("storages", {})
                pushtool = push.get_instance(domain=server)

                if not storages or type(storages) != dict:
                    logger.error(f"[CMD] invalid persist config file '{filepath}'")
                    return

                for k, v in storages.items():
                    if k not in ["modified", "sites"] or not pushtool.validate(v):
                        logger.error(f"[CMD] found invalid configuration '{k}' for server: {server}")
                        return

                params["persist"] = storages
                os.environ["COLLECT_CONF"] = server
        except:
            logger.error(f"[CMD] illegal configuration, must be JSON file")
            return

    os.environ["LOCAL_MODE"] = "true"

    sites = nextweb.collect(params=params)
    if sites and args.backup:
        directory = utils.trim(args.directory)
        if not directory:
            directory = os.path.join(utils.PATH, "data", repository.lower())
        else:
            directory = os.path.abspath(directory)

        # create directory if not exist
        os.makedirs(directory, exist_ok=True)

        filename = utils.trim(args.filename)
        if not filename:
            model = utils.trim(args.model) or "gpt-3.5-turbo"
            now = datetime.now().strftime("%Y%m%d%H%M")
            filename = f"sites-{model}-{now}.txt"

        filepath = os.path.join(directory, filename)
        utils.write_file(filename=filepath, lines=sites, overwrite=True)

        logger.info(f"[CMD] collect finished, {len(sites)} sites have been saved to {filepath}")


if __name__ == "__main__":
    utils.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--async",
        dest="run_async",
        action="store_true",
        default=False,
        help="run with asynchronous mode",
    )

    parser.add_argument(
        "-b",
        "--backup",
        dest="backup",
        action="store_true",
        default=False,
        help="backup results to a local file",
    )

    parser.add_argument(
        "-c",
        "--check",
        dest="checkonly",
        action="store_true",
        default=False,
        help="only check exists sites",
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=False,
        default="",
        help="final available API save path",
    )

    parser.add_argument(
        "-e",
        "--exclude",
        type=str,
        required=False,
        default="",
        help="regular expressions for dropped domains",
    )

    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=False,
        default="",
        help="final available API save file name",
    )

    parser.add_argument(
        "-g",
        "--generate",
        dest="generate",
        action="store_true",
        default=False,
        help="re-generate all data from github",
    )

    parser.add_argument(
        "-i",
        "--ignore",
        dest="nocheck",
        action="store_true",
        default=False,
        help="skip check availability if true",
    )

    parser.add_argument(
        "-k",
        "--chunk",
        type=int,
        required=False,
        default=512,
        help="chunk size of each slice",
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
        help="number of concurrent threads, default twice the number of CPU",
    )

    parser.add_argument(
        "-o",
        "--overlay",
        dest="overlay",
        action="store_true",
        default=False,
        help="append local existing deployments",
    )

    parser.add_argument(
        "-p",
        "--persist",
        type=str,
        required=False,
        default="",
        help="json file for persist config",
    )

    parser.add_argument(
        "-r",
        "--repository",
        type=str,
        required=False,
        default="ChatGPT-Next-Web",
        help="github repository name",
    )

    parser.add_argument(
        "-s",
        "--sort",
        type=str,
        required=False,
        default="newest",
        choices=["newest", "oldest", "stargazers", "watchers"],
        help="forks sort type, see: https://docs.github.com/en/rest/repos/forks",
    )

    parser.add_argument(
        "-t",
        "--typical",
        dest="typical",
        action="store_true",
        default=False,
        help="is a standard OpenAI API subpath",
    )

    parser.add_argument(
        "-u",
        "--username",
        type=str,
        required=False,
        default="ChatGPTNextWeb",
        help="github username",
    )

    main(parser.parse_args())

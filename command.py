import argparse
import json
import os

import push
import utils
from logger import logger
from scripts import nextweb


def main(args: argparse.Namespace) -> None:
    params = {
        "sort": args.sort,
        "refresh": args.refresh,
        "checkonly": args.checkonly,
        "overlay": args.overlay,
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
                        logger.error(
                            f"[CMD] found invalid configuration '{k}' for server: {server}"
                        )
                        return

                params["persist"] = storages
                os.environ["SUBSCRIBE_CONF"] = server
        except:
            logger.error(f"[CMD] illegal configuration, must be JSON file")
            return

    os.environ["LOCAL_MODE"] = "true"
    return nextweb.collect(params=params)


if __name__ == "__main__":
    utils.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--check",
        dest="checkonly",
        action="store_true",
        default=False,
        help="only check exists sites",
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
        "--refresh",
        dest="refresh",
        action="store_true",
        default=False,
        help="re-generate all data from github",
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

    main(parser.parse_args())

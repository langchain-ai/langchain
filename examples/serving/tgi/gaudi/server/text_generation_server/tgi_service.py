import os
from pathlib import Path
from loguru import logger
import sys
from text_generation_server import server
import argparse


def main(args):
    logger.info("TGIService: starting tgi service .... ")
    logger.info(
        "TGIService: --model_id {}, --revision {}, --sharded {}, --dtype {}, --uds_path {} ".format(
            args.model_id, args.revision, args.sharded, args.dtype, args.uds_path
        )
    )
    server.serve(
        model_id=args.model_id, revision=args.revision, dtype=args.dtype, uds_path=args.uds_path, sharded=args.sharded
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--revision", type=str)
    parser.add_argument("--sharded", type=bool)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--uds_path", type=Path)
    args = parser.parse_args()
    main(args)

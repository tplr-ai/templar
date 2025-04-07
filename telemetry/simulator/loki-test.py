import argparse
import asyncio

import bittensor as bt

import tplr


def config() -> bt.Config:
    """
    Parse command-line arguments and return a configuration object.
    """

    parser = argparse.ArgumentParser(
        description="Loki_Test",
        add_help=True,
    )

    bt.subtensor.add_args(parser)
    parser.parse_args()
    return bt.config(parser)


class Loki_Test:
    def __init__(self) -> None:
        self.config = config()
        try:
            version = tplr.__version__
            tplr.logger = tplr.setup_loki_logger(
                service="miner", uid=str("999"), version=version
            )
            tplr.logger.info("Loki logging enabled for miner UID: 999")
        except Exception as e:
            tplr.logger.warning(f"Failed to initialize Loki logging: {e}")

    async def run(self) -> None:
        tplr.logger.info("this is a test")
        await asyncio.sleep(1)
        tplr.logger.error("this is a test")
        await asyncio.sleep(1)
        tplr.logger.warning("this is a test")
        await asyncio.sleep(1)
        tplr.logger.debug("this is a test")
        await asyncio.sleep(1)
        tplr.logger.info("this is a test")
        await asyncio.sleep(1)
        tplr.logger.info("this is a test")
        await asyncio.sleep(20)


def main() -> None:
    loki_test = Loki_Test()
    try:
        asyncio.run(loki_test.run())
    except Exception as e:
        tplr.logger.error(f"loki_test terminated with error: {e}")


if __name__ == "__main__":
    main()

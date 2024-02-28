import argparse
import os
import logging
import sys
import datetime as dt

from code_duality.config import Config
from code_duality.factories import MetricsFactory
from code_duality.metrics import Progress, MemoryCheck, Checkpoint

configs = {
    "debug": "configs/debug.json",
    "mi-vs-algorithms": "configs/mi-vs-algorithms.json",
    "timestep-duality": "configs/timestep-duality.json",
    "coupling-duality-top": "configs/coupling-duality-top.json",
    "coupling-duality-bottom": "configs/coupling-duality-bottom.json",
}


def main(metaconfig: Config, resume: bool = True, save_patience: int = 1):
    metrics = {m: MetricsFactory.build(m) for m in metaconfig.metrics if m.name in MetricsFactory.options()}
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    begin = dt.datetime.now()

    for k, m in metrics.items():
        logger.info(f"---Computing {metrics[k].__class__.__name__}---")
        config = metaconfig.copy().unlock().unlock_types()
        config.metrics = metaconfig.metrics.get(k)

        callbacks = [
            Progress.to_setup(
                logger=logger,
                total=len(config) // config.get("n_async_jobs", 1),
            ),
            MemoryCheck.to_setup("gb", logger=logger),
            Checkpoint.to_setup(
                patience=save_patience,
                savepath=config.path,
                logger=logger,
                metrics=metrics[k],
            ),
        ]
        m.compute(
            config, resume=resume, callbacks=callbacks, n_workers=config.n_workers, n_async_jobs=config.n_async_jobs
        )
        for c in callbacks:
            c.teardown()

    end = dt.datetime.now()
    logger.info(f"Total computation time: {end - begin}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="The configuration name or file.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="The number of available workers.",
    )
    parser.add_argument(
        "--n-async-jobs",
        type=int,
        default=1,
        help="The number of asynchronous jobs to use.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./",
        help="The output path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the computation.",
    )
    parser.add_argument(
        "--save-patience",
        type=int,
        default=1,
        help="Patience for saving.",
    )

    args = parser.parse_args()

    # Preparing the config
    if args.config in configs:
        config = Config.load(configs[args.config]).unlock()
    else:
        assert os.path.exists(args.config), f"Config file {args.config} not found."
        config = Config.load(args.config).unlock()
    config.n_workers = args.n_workers
    config.n_async_jobs = args.n_async_jobs
    config.path = args.output_path
    config.lock()

    # Running the main program
    main(config, resume=args.resume, save_patience=args.save_patience)

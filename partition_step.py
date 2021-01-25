import argparse
from azureml.core.run import Run
import logging
from math import ceil
from opencensus.ext.azure.log_exporter import AzureLogHandler
import pandas as pd
from pathlib import Path
import time
from typing import Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
logger.addHandler(AzureLogHandler())

run = Run.get_context(allow_offline=False)

logging_metadata = {'custom_dimensions': {
    'parent_run_id': run.parent.id,
    'step_id': run.id,
    'step_name': run.name,
    'experiment_name': run.experiment.name,
    'run_url': run.parent.get_portal_url(),
}}


def debug(msg):
    logger.debug(msg, extra=logging_metadata)


def info(msg):
    logger.info(msg, extra=logging_metadata)


def warning(msg):
    logger.warning(msg, extra=logging_metadata)


def error(msg):
    logger.error(msg, extra=logging_metadata)


def exception(msg):
    logger.exception(msg, extra=logging_metadata)


def load_data(path):
    files = list(path.glob(f'**/*.parquet'))
    info(f'Reading input files: {files}')
    df = pd.concat([pd.read_parquet(file) for file in files], verify_integrity=True, copy=False)
    info(f'Loaded {len(df)} rows.')
    return df


def partition_data_and_write_output_files(df, output_path: Path):
    chunk_size = 1000
    output_path.mkdir(parents=True, exist_ok=True)
    num_chunks = ceil(len(df) / chunk_size)
    num_chunks = min(num_chunks, 10)  # cut down to 10 for testing purposes
    tic = time.time()
    for i in range(num_chunks):
        df.iloc[i * chunk_size:(i + 1) * chunk_size, :]\
            .to_parquet(output_path / Path(f'raw-{i:05d}.parquet'), engine='pyarrow', index=True)
        if i % (100000 // chunk_size) == 0:
            dt = time.time() - tic
            debug(f'Wrote {(i + 1) * chunk_size} items in {i + 1} chunks in {dt:.0f} sec.')
            debug(f'That is on average {(i + 1) * chunk_size / dt:.3f} items per second.')
    info(f'Partitioned raw data into {num_chunks} chunks.')


def parse_arguments() -> Tuple[Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', dest='output_dir', type=str, required=True)
    args, _ = parser.parse_known_args()
    input_path = Path(run.input_datasets['ds_raw'])  # noqa
    output_path = Path(args.output_dir)
    return input_path, output_path


def main():
    input_path, output_path = parse_arguments()
    df = load_data(input_path)
    partition_data_and_write_output_files(df, output_path)


if __name__ == '__main__':
    main()

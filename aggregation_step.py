import argparse
from azureml.core.run import Run
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler
import pandas as pd
from pathlib import Path
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


def read_input(input_path):
    input_files = list(input_path.glob(f'**/*.parquet'))
    info(f'Reading input files: {input_files}')
    df = pd.concat([pd.read_parquet(file) for file in input_files], ignore_index=True, copy=False)
    info(f'Loaded {len(df)} rows.')
    return df


def write_output(df, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'aggregated.txt', 'w') as f:
        f.write('\n'.join(df['extracted-objects'].dropna().unique()))


def parse_arguments() -> Tuple[Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', dest='output_dir', type=str, required=True)
    args, _ = parser.parse_known_args()
    input_path = Path(run.input_datasets['ds_processed'])  # noqa
    output_path = Path(args.output_dir)
    return input_path, output_path


def main():
    input_path, output_path = parse_arguments()
    debug('Reading input')
    df_input = read_input(input_path)
    debug('Writing output')
    write_output(df_input, output_path)
    debug('Done')


if __name__ == '__main__':
    main()

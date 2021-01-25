import argparse
from azureml.core.run import Run
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler
import pandas as pd
import parse
from pathlib import Path
import time
from typing import Optional, Sequence


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
logger.addHandler(AzureLogHandler())

run_ = Run.get_context(allow_offline=False)

logging_metadata = {'custom_dimensions': {
    'parent_run_id': run_.parent.id,
    'step_id': run_.id,
    'step_name': run_.name,
    'experiment_name': run_.experiment.name,
    'run_url': run_.parent.get_portal_url(),
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


input_path: Optional[Path] = None
output_path: Optional[Path] = None


def init():

    global input_path, output_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', dest='output_dir', type=str, required=True)
    args, _ = parser.parse_known_args()
    input_path = Path(run_.input_datasets['ds_partitioned'])  # noqa
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    debug('Processor initialized')


def run(data_files):
    debug(f'Received mini-batch of size {len(data_files)}: {str(data_files)}')
    for file in data_files:
        partition_nr = get_partition_nr(file)
        df = pd.read_parquet(file, engine='pyarrow')
        df = df[:10]
        info(f'{partition_nr:05d}: Loaded {len(df)} documents.')
        extracted_data = perform_extraction(df)
        info(f'{partition_nr:05d}: Extraction performed on {len(extracted_data)} documents.')
        store_results(extracted_data, partition_nr)
    # output is ignored since the step is executed with output_action='summary_only'
    return ['Success' for _ in data_files]


def get_partition_nr(path):
    parse_result = parse.search('raw-{partition_nr:d}.parquet', Path(path).name)
    return parse_result['partition_nr']


def perform_extraction(df) -> Sequence[dict]:
    tic = time.time()
    tic_last = tic
    extracted_data = []
    for docid, row in df.iterrows():
        extracted_data.append({
            'docid': str(docid),
            'extracted-objects': row['fulltext'][:10]
        })
        t_now = time.time()
        if t_now - tic_last > 60:
            dt = t_now - tic
            debug(f'Processed {len(extracted_data)} documents in {dt:.0f} sec. '
                  f'That is {dt / len(extracted_data):.3f} sec per document on average.')
            tic_last = t_now
    return extracted_data


def store_results(extracted_data: Sequence[dict], partition_nr):

    global output_path

    output_file_name = Path(f'extracted-data-{partition_nr:05d}.parquet')
    output_file_path = output_path / output_file_name
    debug(f'{partition_nr:05d}: Writing to {output_file_path}')

    result = {
        'docid': [],
        'extracted-objects': [],
    }

    n_docs = len(extracted_data)
    for i_doc in range(n_docs):
        result['docid'].append(extracted_data[i_doc]['docid'])
        result['extracted-objects'].append(extracted_data[i_doc]['extracted-objects'])

    df = pd.DataFrame.from_dict(data=result)

    df.to_parquet(output_file_path, engine='pyarrow', index=True)
    debug(f'{partition_nr:05d}: Written to {output_file_path}')

import json
import os
import datasets


def there_exist_parquet_files_in_this_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            return True
    return False


def load_dataset_info(directory):
    with open(os.path.join(directory, "dataset_info.json"), "r") as f:
        return datasets.DatasetInfo.from_dict(json.load(f))


def dataset_info_says_buider_is_parquet(directory):
    info = load_dataset_info(directory)
    return info.builder_name == "parquet"


def find_parquet_files_for_split(directory, split):
    for filename in os.listdir(directory):
        if filename.endswith(".parquet") and filename.startswith(split + "-"):
            yield os.path.join(directory, filename)


def load_dataset(path, *args, **kwargs):
    """
    Extends datasets.load_dataset to support loading from parquet files.

    # TODO: it currently downloads the parquet files to the tmp folder.
    e.g. Downloading and preparing dataset parquet/default to /root/.cache/huggingface/datasets/parquet
    This is not necessary.
    """
    if there_exist_parquet_files_in_this_directory(path):
        if dataset_info_says_buider_is_parquet(path):
            dataset_info = load_dataset_info(path)
            data_files = {}
            for split in dataset_info.splits:
                data_files[split] = sorted(
                    list(find_parquet_files_for_split(path, split))
                )
            return datasets.load_dataset(
                "parquet", data_files=data_files, *args, **kwargs
            )
    return datasets.load_dataset(*args, **kwargs)

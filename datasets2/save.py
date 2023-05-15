import tqdm
import itertools
import os
from datasets.arrow_dataset import Dataset
from datasets.splits import SplitDict, SplitInfo
from datasets.features.features import require_decoding
from datasets.utils.py_utils import convert_file_size_to_int
from datasets import config as datasets_config
from datasets.table import embed_table_storage

# TODO (future): consider saving a dataset script to the folder.
#    Example: https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py


def save_to_disk(
    dataset: Dataset,
    outdir: str,
    parquet: bool = False,
    num_shards=2,
    embed_external_files: bool = True,
    dataset_name: str = None,
    **kwargs,
):
    """
    Extends datasets.Dataset.save_to_disk to support saving to parquet files.

    dataset: A dataset that is already splitted such that you can call ds["train"], ds["test"] etc.
    outdir: The directory to save the dataset to.
    parquet: If True, save the dataset as parquet files.
    num_shards: The number of shards to save the dataset to.
    embed_external_files: If True, embed external files into the dataset.
    dataset_name: The name of the dataset. If None, use the name of the dataset.

    Reference:
        https://github.com/huggingface/datasets/src/datasets/arrow_dataset.py
    """
    if not parquet:
        return dataset.save_to_disk(outdir, **kwargs)
    infos = []
    for split in dataset.keys():
        subset = dataset[split]
        info = save_as_parquet(
            subset,
            split=split,
            outdir=outdir,
            num_shards=num_shards,
        )
        infos.append(info)

    # Merge the infos.
    merged_info = infos[0]
    dataset_sizes = [i.dataset_size for i in infos]
    dataset_sizes = [x for x in dataset_sizes if x is not None]
    sizes_in_bytes = [i.size_in_bytes for i in infos]
    sizes_in_bytes = [x for x in sizes_in_bytes if x is not None]
    split_infos = [i.splits for i in infos]
    data_files = []
    for i in infos:
        data_files.extend(i.data_files)
    if len(dataset_sizes) > 0:
        merged_info.dataset_size = sum(dataset_sizes)
    if len(sizes_in_bytes) > 0:
        merged_info.size_in_bytes = sum(sizes_in_bytes)
    splits = {}
    for split_info in split_infos:
        for k, v in split_info.items():
            splits[k] = v
    merged_info.splits = splits
    # merged_info.data_files = data_files  # This is not saved anyway.
    merged_info.builder_name = "parquet"
    merged_info.write_to_directory(outdir)


def save_as_parquet(
    ds: Dataset,  # should not contain subsplits
    outdir: str,
    split: str = "train",
    num_shards=2,
    embed_external_files: bool = True,
    dataset_name: str = None,
):
    os.makedirs(outdir, exist_ok=True)
    # Find decodable columns, because if there are any, we need to:
    # embed the bytes from the files in the shards
    decodable_columns = (
        [
            k
            for k, v in ds._info.features.items()
            if require_decoding(v, ignore_decode_attribute=True)
        ]
        if embed_external_files
        else []
    )

    if hasattr(ds, "_estimate_nbytes"):
        dataset_nbytes = ds._estimate_nbytes()

        if num_shards is None:
            max_shard_size = convert_file_size_to_int(
                max_shard_size or datasets_config.MAX_SHARD_SIZE
            )
            num_shards = int(dataset_nbytes / max_shard_size) + 1
            num_shards = max(num_shards, 1)
    else:
        dataset_nbytes = None
        if num_shards is None:
            num_shards = 1

    shards = (
        ds.shard(num_shards=num_shards, index=i, contiguous=True)
        for i in range(num_shards)
    )

    if decodable_columns:

        def shards_with_embedded_external_files(shards):
            for shard in shards:
                shard_format = shard.format
                shard = shard.with_format("arrow")
                shard = shard.map(
                    embed_table_storage,
                    batched=True,
                    batch_size=1000,
                    keep_in_memory=True,
                )
                shard = shard.with_format(**shard_format)
                yield shard

        shards = shards_with_embedded_external_files(shards)

    def path_in_repo(_index, shard):
        return f"{outdir}/{split}-{_index:05d}-of-{num_shards:05d}-{shard._fingerprint}.parquet"

    shards_iter = iter(shards)
    first_shard = next(shards_iter)

    shards_path_in_repo = []
    for index, shard in tqdm.tqdm(
        enumerate(itertools.chain([first_shard], shards_iter)),
        desc="preparing dataset shards",
        total=num_shards,
        # disable=not logging.is_progress_bar_enabled(),
    ):
        shard_path_in_repo = path_in_repo(index, shard)
        # print("")
        # print(shard_path_in_repo)
        # print(type(shard))
        shard.to_parquet(shard_path_in_repo)
        shards_path_in_repo.append(str(shard_path_in_repo))

    # Populate dataset info.
    info_to_dump = ds.info.copy()
    info_to_dump.dataset_size = dataset_nbytes
    info_to_dump.size_in_bytes = dataset_nbytes
    info_to_dump.data_files = shards_path_in_repo
    info_to_dump.splits = SplitDict(
        {
            split: SplitInfo(
                split,
                num_bytes=dataset_nbytes,
                num_examples=len(ds),
                dataset_name=dataset_name,
            )
        }
    )

    return info_to_dump

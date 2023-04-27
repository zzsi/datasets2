from datasets import list_datasets, load_dataset, load_from_disk
from datasets.features.features import require_decoding
from datasets.utils.py_utils import convert_file_size_to_int
from datasets import config as datasets_config
from datasets.table import embed_table_storage
import tqdm
import itertools
from io import BytesIO


import os


def save_as_parquet(
    ds, outdir: str, split: str = "train", num_shards=2, embed_external_files=True
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

    dataset_nbytes = ds._estimate_nbytes()

    if num_shards is None:
        max_shard_size = convert_file_size_to_int(
            max_shard_size or datasets_config.MAX_SHARD_SIZE
        )
        num_shards = int(dataset_nbytes / max_shard_size) + 1
        num_shards = max(num_shards, 1)

    shards = (
        ds.shard(num_shards=num_shards, index=i, contiguous=True)
        for i in range(num_shards)
    )

    if decodable_columns:

        def shards_with_embedded_external_files(shards):
            for shard in shards:
                format = shard.format
                shard = shard.with_format("arrow")
                shard = shard.map(
                    embed_table_storage,
                    batched=True,
                    batch_size=1000,
                    keep_in_memory=True,
                )
                shard = shard.with_format(**format)
                yield shard

        shards = shards_with_embedded_external_files(shards)

    def path_in_repo(_index, shard):
        return f"{outdir}/{split}-{_index:05d}-of-{num_shards:05d}-{shard._fingerprint}.parquet"

    shards_iter = iter(shards)
    first_shard = next(shards_iter)

    uploaded_size = 0
    shards_path_in_repo = []
    for index, shard in tqdm.tqdm(
        enumerate(itertools.chain([first_shard], shards_iter)),
        desc="preparing dataset shards",
        total=num_shards,
        # disable=not logging.is_progress_bar_enabled(),
    ):
        shard_path_in_repo = path_in_repo(index, shard)
        print("")
        print(shard_path_in_repo)
        print(type(shard))
        shard.to_parquet(shard_path_in_repo)
        shards_path_in_repo.append(str(shard_path_in_repo))

    return shards_path_in_repo

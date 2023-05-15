`datasets2` provides add-ons to the huggingface `datasets`.


## Install

```
pip install datasets2
```

# Example usage

```
# datasets is just the huggingface datasets
# load_dataset and save_to_disk adds parquet support to datasets.load_dataset and datasets.Dataset.save_to_disk
from datasets2 import datasets, load_dataset, save_to_disk

my_dataset = prepare_my_huggingface_dataset()
output_dir = "my_dataset_dir"
# num_shards controls how many parquet files to use
save_to_disk(my_dataset, output_dir, parquet=True, num_shards=2)

load_dataset(output_dir)  # automatically infers if the dataset uses parquet format.

```

If you want `save_to_disk` to behave the same as how the original `datasets` saves data, just call it without the `parquet` argument:

```
save_to_disk(my_dataset, output_dir)
```


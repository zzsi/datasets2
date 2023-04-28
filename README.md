`datasets2` provides add-ons to the huggingface `datasets`.

Example usage:

```
# datasets is just the huggingface datasets
# load_dataset and save_to_disk adds parquet support to datasets.load_dataset and datasets.Dataset.save_to_disk
from datasets2 import datasets, load_dataset, save_to_disk

my_dataset = prepare_my_huggingface_dataset()
output_dir = "my_dataset_dir"
save_to_disk(my_dataset, output_dir, parquet=True)

load_dataset(output_dir)  # automatically infers if the dataset uses parquet format.

```


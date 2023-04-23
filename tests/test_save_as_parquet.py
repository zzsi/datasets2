import json
from pathlib import Path
from datasets2.save import save_as_parquet


def prepare_jsonline_and_jpg_files(data_dir: str):
    # The original format includes numpy arrays.
    # Convert it to a set of jpg files which conforms with how
    # we often store image datasets.
    mnist = load_dataset("mnist")
    test_ds = mnist["test"]
    rows = []
    data_dir = Path(data_dir)
    for i, row in enumerate(test_ds):
        fn = str(data_dir / f"images/{i:05d}.jpg")
        row["image"].save(fn)
        rows.append({"image": fn})
        if i >= 100:
            break

    metadata_path = data_dir / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")

    ds = load_dataset("json", data_files=["metadata.jsonl"])
    # cast string to Image
    ds_casted = ds.cast_column("image", Image(decode=True))
    return ds_casted


def test_saving_mnist_as_parquet(tmpdir):
    jsonline_and_image_files_dir = tmpdir.mkdir("source_data")
    parquet_dir = tmpdir.mkdir("parquet_data")
    ds = prepare_jsonline_and_jpg_files(str(jsonline_and_image_files_dir))
    first_example = ds["train"][0]
    save_as_parquet(ds["train"], outdir=str(parquet_dir))

    # TODO: Now remove the source data, and then load from parquet.
    # Verify the the image data is still loaded by checking the first
    # example.

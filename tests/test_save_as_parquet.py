import json
import numpy as np
from pathlib import Path
import pytest
from PIL import Image as PILImage
import shutil

import datasets
from datasets.features import Image
from datasets2.save import save_as_parquet, save_to_disk
from datasets2.load import load_dataset


def prepare_jsonline_and_jpg_files(data_dir: str):
    # The original format includes numpy arrays.
    # Convert it to a set of jpg files which conforms with how
    # we often store image datasets.
    rows = []
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(100):
        fn0 = data_dir / f"images/{i:05d}.png"
        # create dir
        fn0.parent.mkdir(parents=True, exist_ok=True)
        fn = str(fn0)
        # create a random image of size 28, 28
        img = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
        pil_img = PILImage.fromarray(img)
        pil_img.save(fn)
        rows.append({"image": fn})
        if i >= 100:
            break

    metadata_path = data_dir / "metadata.jsonl"
    with open(str(metadata_path), "w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")

    ds = datasets.load_dataset("json", data_files=[str(metadata_path)])
    # cast string to Image
    ds_casted = ds.cast_column("image", Image(decode=True))
    return ds_casted


def _test_saving_mnist_as_parquet(tmpdir):
    jsonline_and_image_files_dir = tmpdir.mkdir("source_data")
    parquet_dir = Path(str(tmpdir)) / "parquet_data"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    ds = prepare_jsonline_and_jpg_files(str(jsonline_and_image_files_dir))
    assert list(ds.keys()) == ["train"]
    first_example = ds["train"][0]
    img = first_example["image"]
    np_img = np.array(img)
    info = save_as_parquet(ds["train"], outdir=str(parquet_dir))
    info.write_to_directory(str(parquet_dir))

    # Now remove the source data, and then load from parquet.
    # Verify the the image data is still loaded by checking the first
    # example.
    shutil.rmtree(str(jsonline_and_image_files_dir))
    # load parquet files
    parquet_ds = datasets.load_dataset(
        "parquet",
        data_files=info.data_files,
    )
    first_parquet_example = parquet_ds["train"][0]
    parquet_np_img = np.array(first_parquet_example["image"])

    assert np.array_equal(np_img, parquet_np_img)


@pytest.mark.dev
def test_simple_saving_and_loading_parquet(tmpdir):
    jsonline_and_image_files_dir = tmpdir.mkdir("source_data")
    parquet_dir = Path(str(tmpdir)) / "parquet_data"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    ds = prepare_jsonline_and_jpg_files(str(jsonline_and_image_files_dir))
    first_example = ds["train"][0]
    img = first_example["image"]
    np_img = np.array(img)
    save_to_disk(ds, outdir=str(parquet_dir), parquet=True)

    shutil.rmtree(str(jsonline_and_image_files_dir))

    info_path = parquet_dir / "dataset_info.json"
    assert info_path.exists()
    with open(str(info_path), encoding="utf-8") as f:
        info = json.load(f)
        assert info["dataset_size"] == 88413
        print(info["splits"])
        assert "data_files" not in info
        # assert info == None, f"{info.keys()}"
        assert info["size_in_bytes"] == 88413
        assert info["builder_name"] == "parquet"

    # load parquet files
    parquet_ds = load_dataset(str(parquet_dir))
    assert list(parquet_ds.keys()) == ["train"]
    assert len(parquet_ds["train"]) == 100
    first_parquet_example = parquet_ds["train"][0]
    parquet_np_img = np.array(first_parquet_example["image"])
    assert np.array_equal(
        np_img, parquet_np_img
    ), f"{np_img.ravel()[:3]} != {parquet_np_img.ravel()[:3]}"

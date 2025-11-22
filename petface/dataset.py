import pathlib
import polars as pl
import subprocess

from tempfile import NamedTemporaryFile


DATASET_DIR = pathlib.Path(__file__).parent.parent / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
SPLIT_DIR = DATASET_DIR / "split"
OUT_DIR = DATASET_DIR / "out"


def move_files(
    src_dir: pathlib.Path,
    dst_dir: pathlib.Path,
    file_paths_to_include: list[str]
) -> None:
    with NamedTemporaryFile("w") as t_file:
        file_paths_to_include_str = "\n".join(file_paths_to_include)
        with open(t_file.name, "w") as f:
            f.write(file_paths_to_include_str)

        cmd = [
            "rclone",
            "move",
            str(src_dir),
            str(dst_dir),
            "--include-from",
            t_file.name,
            "--progress",
            "--delete-empty-src-dirs"
        ]

        print(f'Running command: {" ".join(cmd)}')

        subprocess.run(cmd, check=True)


def build_reidentification_train_test_split(animal: str) -> None:
    animal_split_dir = SPLIT_DIR / animal

    train_file = animal_split_dir / "train.csv"
    reidentification_file = animal_split_dir / "reidentification.csv"
    
    train_set = pl.read_csv(train_file)
    reidentification_set = pl.read_csv(reidentification_file)

    train_file_paths = train_set["filename"].to_list()
    reidentification_file_paths = reidentification_set["filename"].to_list()

    train_dir = OUT_DIR / "images" / "train"
    test_dir = OUT_DIR / "images" / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    move_files(
        src_dir=IMAGES_DIR,
        dst_dir=train_dir,
        file_paths_to_include=train_file_paths
    )

    move_files(
        src_dir=IMAGES_DIR,
        dst_dir=test_dir,
        file_paths_to_include=reidentification_file_paths
    )


build_reidentification_train_test_split("dog")

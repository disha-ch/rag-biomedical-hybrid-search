"""Initial dataset inspection script.

Purpose:
- verify the exact Hugging Face dataset structure,
- inspect available splits and columns,
- inspect a few records before defining the normalized schema.

This script intentionally avoids retrieval logic.
"""

from datasets import load_dataset

DATASET_NAME = "rag-datasets/rag-mini-bioasq"


def main() -> None:
    dataset = load_dataset(DATASET_NAME)

    print("Available splits:", list(dataset.keys()))

    for split_name, split in dataset.items():
        print(f"\n--- {split_name} ---")
        print("Rows:", len(split))
        print("Columns:", split.column_names)

        if len(split) > 0:
            print("Sample record:")
            print(split[0])


if __name__ == "__main__":
    main()

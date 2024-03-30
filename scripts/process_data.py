import pandas as pd
import argparse
from src.processor import Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/data_3D.pickle",
        help="Path to the EC3D dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save the preprocessed data",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    dataset = pd.read_pickle(args.data_path)

    processor = Processor(dataset)
    processor.process_data(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)

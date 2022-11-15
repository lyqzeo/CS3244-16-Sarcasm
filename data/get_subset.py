import argparse

import pandas as pd

def get_subset(input_path, output_file):
    """
    Gets a subset of instances from file at input_path and writes to the output_file path

    For desired number of instances, change the variable "N"
    """

    N = 10000 # CHANGE THIS TO THE DESIRED DATASET INSTANCES
    half = int(N / 2) 

    df = pd.read_csv(input_path)

    neg = df.loc[df["label"] == 0].head(half)
    pos = df.loc[df["label"] == 1].head(half)

    res = pd.concat([neg, pos])
    res.to_csv(output_file, index=False)

def main(args):
    assert args.input and args.output, "Please specify --input and --output"
    get_subset(args.input, args.output)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='file to be pre-processed')
    parser.add_argument('--output', help='pre-processed file')
    return parser.parse_args()

# example:
# python get_subset.py --input train-balanced-sarcasm.csv --output balanced.csv
if __name__ == "__main__":
    args = get_arguments()
    main(args)

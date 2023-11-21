# Load a jsonl file and create a new 
# entry called "options" that is a list of
#  k randomly selected label_col values from the
# rest of the file and which includes the original
# "label" value. The new entry is saved to a new
# jsonl file.

from src.utils import read_jsonl, write_jsonl
import random
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_total_options", default=4, type=int, required=False)
    parser.add_argument("--label_col", default="label", type=str, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    args = parser.parse_args()


    random.seed(args.seed)
    data = read_jsonl(args.input_file)
    
    label_col = args.label_col
    for item in data:
        item["options"] = [item[label_col]] + random.sample([x[label_col] for x in data if x[label_col] != item[label_col]], args.num_total_options - 1)
    write_jsonl(data, args.output_file)
    print(list(data[0].keys()))
if __name__ == "__main__":
    main()

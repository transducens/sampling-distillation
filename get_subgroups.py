import sys
import argparse

def split(args) -> None:

    output_file = args.output_file
    size = args.groups

    with open(output_file, "w") as f_output:
        for i, line in enumerate(sys.stdin):
            if i %10 < size:
                f_output.write(line)

def main() -> None:
    parser = argparse.ArgumentParser(description="Split training corpora in subgroups")
    parser.add_argument("--groups", type=int, required=True, help="The size of the sentences groups")
    parser.add_argument("--output-file", type=str, required=True, help="The output file")
    args = parser.parse_args()

    split(args)


if __name__ == "__main__":
    main()

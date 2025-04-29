import argparse
import contextlib
import sys

def remove(src, tgt):

    with open(src) as f:
        source = [line.strip() for line in f]

    with open(tgt) as f:
        target = [line.strip() for line in f]

    source_clean = []
    target_clean = []
    for i, line in enumerate(target):
        if len(line) > 0 and len(source[i]) > 0:
            source_clean.append(source[i])
            target_clean.append(line)
    
    with open(src+"-clean", mode='a', encoding='utf-8') as output_source:
            output_source.write('\n'.join(source_clean) + '\n')
    with open(tgt+"-clean", mode='a', encoding='utf-8') as output_target:
            output_target.write('\n'.join(target_clean) + '\n')



def main() -> None:
    parser = argparse.ArgumentParser(description="Remove empty lines from training corpora")
    parser.add_argument("--src", type=str, required=True, help="The source corpus")
    parser.add_argument("--tgt", type=str, required=True, help="The target corpus")

    args = parser.parse_args()

    remove(src=args.src, tgt=args.tgt)



if __name__ == "__main__":
    main()
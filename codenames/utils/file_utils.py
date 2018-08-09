from typing import List


def read_lines(input_file: str) -> List[str]:
    with open(input_file) as f:
        lines = f.readlines()
        return [l.strip() for l in lines]


def read_lines_tokens(input_file: str) -> List[str]:
    with open(input_file) as f:
        lines = f.readlines()
        return [l.strip().split(' ') for l in lines]

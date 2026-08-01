#!/usr/bin/env python3
import argparse
from pathlib import Path
import v4_temporal_optimizer as v
import v4_fast_driver as driver

_original_encode = v.encode

def encode_without_day(df, columns=None):
    return _original_encode(df.drop(columns=['day'], errors='ignore'), columns)

v.encode = encode_without_day

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    driver.main(args.corpus, args.output)

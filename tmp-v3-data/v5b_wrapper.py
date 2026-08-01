#!/usr/bin/env python3
import argparse
from pathlib import Path
import v5_walkforward_driver as driver

driver.TOPK = [1, 2, 3, 4, 5]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    driver.evaluate(a.corpus, a.output)

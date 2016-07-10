#! /usr/bin/env python

from __future__ import print_function
import sys
import pandas as pd

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s infile.csv outfile.csv' % sys.argv[0],
              file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    df.sort_values(by='time').to_csv(sys.argv[2], index=False)

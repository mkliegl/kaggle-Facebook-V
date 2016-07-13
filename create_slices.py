#! /usr/bin/env python

from __future__ import print_function, division

import os
import sys
import pandas as pd

time_split = 786239 - 10 * 7 * 24 * 60  # use last 10 weeks for validation

# split up grid by making horizontal slices
num_slices = 32
slice_height = 10.0 / num_slices + 1e-6
slice_overlap = 0.1


def gen_slice(df_train, df_test, i):
    slice_start = i * slice_height
    slice_end = (i + 1) * slice_height
    print('Generating slice %d: %.3f to %.3f' % (i, slice_start, slice_end))

    # create train slice
    train_slice = df_train[
        (df_train.y >= slice_start - slice_overlap) &
        (df_train.y < slice_end + slice_overlap)].copy()
    train_slice[['row_id', 'x', 'y', 'accuracy', 'time', 'place_id']].to_csv(
        '%s/s%i_train.csv' % (output_dir, i), index=False)

    # create test slice
    test_slice = df_test[
        (df_test.y >= slice_start) &
        (df_test.y < slice_end)]
    test_slice[['row_id', 'x', 'y', 'accuracy', 'time']].to_csv(
        '%s/s%i_test.csv' % (output_dir, i), index=False)

    # create train & test slices for validation
    # - split based on time
    # - remove unseen place_id's from validation set
    vtrain_slice = train_slice[train_slice.time <= time_split]
    vtrain_slice[['row_id', 'x', 'y', 'accuracy', 'time', 'place_id']].to_csv(
        '%s/s%i_vtrain.csv' % (output_dir, i), index=False)

    place_ids = set(vtrain_slice.place_id.unique())
    vtest_slice = train_slice[
        (train_slice.y >= slice_start) &
        (train_slice.y < slice_end) &
        (train_slice.time > time_split) &
        (train_slice.place_id.isin(place_ids))]
    vtest_slice[['row_id', 'x', 'y', 'accuracy', 'time', 'place_id']].to_csv(
        '%s/s%i_vtest.csv' % (output_dir, i), index=False)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Expected 3 arguments: train_filename test_filename output_dir",
              file=sys.stderr)
        sys.exit(1)

    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    output_dir = sys.argv[3]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_train = pd.read_csv(train_filename)
    df_test = pd.read_csv(test_filename)

    for i in range(num_slices):
        gen_slice(df_train, df_test, i)

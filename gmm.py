#! /usr/bin/env python
"""Compute GMM's for (x, y) distribution of each place.

Removes outliers and tries fitting GMM's up to a specified maximum number of
components for each place_id. The number of components giving the best AIC
score is used.

Usage: ./gmm.py infile.csv outfile.csv
"""

from __future__ import print_function, division

import sys
import time

import numpy as np
import pandas as pd
from sklearn.mixture import GMM

random_state = 42

x_cutoff = None   # remove points this far from median (x, y)
y_cutoff = 0.03   # location of checkins for a given place
max_comp = 15     # maximum number of GMM components to fit
min_covar = 1e-5  # minimum x, y covariance


def place_iterator(df, places, limit=None):
    for place_id in places[:limit]:
        yield df[df.place_id == int(place_id)].reset_index()


def remove_outliers(x, y, x_cutoff=None, y_cutoff=None):
    if (x_cutoff is None) and (y_cutoff is None):
        return x, y
    if x_cutoff is None:
        x_cutoff = np.float('inf')
    if y_cutoff is None:
        y_cutoff = np.float('inf')
    x_med = x.median()
    y_med = y.median()
    idxs = ((x - x_med < x_cutoff) & (x_med - x < x_cutoff) &
            (y - y_med < y_cutoff) & (y_med - y < y_cutoff))
    return x[idxs], y[idxs]


def compute_gmms(
        df, places, max_comp=12,
        x_cutoff=None, y_cutoff=None, limit=None, min_covar=1e-5):
    start_time = time.time()
    res = np.zeros((len(places[:limit]), 3 + 5 * max_comp), dtype=object)
    for i, dfp in enumerate(place_iterator(df, places, limit)):
        if i % 100 == 0:
            print('%d: %.2fs' % (i, time.time() - start_time))
        xs, ys = remove_outliers(dfp.x, dfp.y, x_cutoff, y_cutoff)
        data = np.concatenate(
            (xs.reshape((-1, 1)), ys.reshape((-1, 1))), axis=1)
        num_points = len(data)
        best_aic, best_gmm = float('inf'), None
        if num_points >= 2:  # GMM needs at least 2 points
            max_n_comp = min(num_points, max_comp)
            for n_comp in range(1, 1 + max_n_comp):
                gmm = GMM(
                    n_components=n_comp,
                    covariance_type='diag',
                    min_covar=min_covar,
                    random_state=random_state)
                gmm.fit(data)
                aic = gmm.aic(data)
                if aic < best_aic:
                    best_aic, best_gmm = aic, gmm
        res[i, 0] = dfp.place_id[0]
        res[i, 1] = num_points
        if best_gmm is not None:
            res[i, 2] = best_gmm.n_components
            for j in range(best_gmm.n_components):
                res[i, 3 + 5 * j] = best_gmm.weights_[j]
                res[i, 4 + 5 * j] = best_gmm.means_[j, 0]
                res[i, 5 + 5 * j] = best_gmm.means_[j, 1]
                res[i, 6 + 5 * j] = best_gmm.covars_[j, 0]
                res[i, 7 + 5 * j] = best_gmm.covars_[j, 1]
    col_names = ['place_id', 'n_pts', 'n_components']
    for j in range(max_comp):
        col_names += [
            'weight_%d' % j,
            'mean_x_%d' % j,
            'mean_y_%d' % j,
            'var_x_%d' % j,
            'var_y_%d' % j,
        ]
    return pd.DataFrame(res, columns=col_names)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s infile.csv outfile.csv' % sys.argv[0],
              file=sys.stderr)
        sys.exit(1)
    df_train = pd.read_csv(sys.argv[1])
    places = df_train.place_id.unique()
    print('%d places' % len(places))
    res = compute_gmms(
        df_train, places, limit=None,
        max_comp=max_comp, min_covar=min_covar,
        x_cutoff=x_cutoff, y_cutoff=y_cutoff)
    res.to_csv(sys.argv[2], index=False)

import numpy as np
import pytest

from sklearn.utils.testing import assert_array_equal

from scipy.sparse import bsr_matrix, csc_matrix, csr_matrix

from sklearn.feature_selection import CorrelationThreshold

data = [[0, 1, 2, 3, 4],
        [0, 2, 2, 3, 5],
        [1, 1, 2, 4, 0]]

y = [[2],[4], [2]]


def test_one_correlation():
    # Test CorrelationThreshold with default setting, correlation = 1.

    for X in [data]:
        sel = CorrelationThreshold().fit(X)
        print(sel)
        assert_array_equal([0, 1, 4], sel.get_support(indices=True))


def test_correlation_threshold():
    # Test VarianceThreshold with custom variance.
    for X in [data]:
        X_s = CorrelationThreshold(threshold=.4).fit_transform(X)
        assert (len(data), 1) == X_s.shape
        X_s = CorrelationThreshold(threshold=.6).fit_transform(X)
        assert (len(data), 2) == X_s.shape
        X_s = CorrelationThreshold(n_features=3).fit_transform(X)
        assert (len(data), 3) == X_s.shape


def test_max_relevance():
    # Test if we select the highest correlated column first:
    sel = CorrelationThreshold(threshold=.4, n_features=1).fit(data, y)
    X_s = sel.transform(data)
    print(X_s)
    assert (len(data), 1) == X_s.shape
    assert_array_equal([1], sel.get_support(indices=True))

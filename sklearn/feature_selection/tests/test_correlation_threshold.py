import numpy as np
import pytest

from sklearn.utils.testing import assert_array_equal

from scipy.sparse import bsr_matrix, csc_matrix, csr_matrix

from sklearn.feature_selection import CorrelationThreshold

data = [[0, 1, 2, 3, 4],
        [0, 2, 2, 3, 5],
        [1, 1, 2, 4, 0]]


def test_zero_variance():
    # Test VarianceThreshold with default setting, zero variance.

    for X in [data, csr_matrix(data), csc_matrix(data), bsr_matrix(data)]:
        sel = CorrelationThreshold().fit(X)
        print(sel)
        assert_array_equal([0, 1, 3, 4], sel.get_support(indices=True))

    with pytest.raises(ValueError):
        VarianceThreshold().fit([[0, 1, 2, 3]])
    with pytest.raises(ValueError):
        VarianceThreshold().fit([[0, 1], [0, 1]])


def test_correlation_threshold():
    # Test VarianceThreshold with custom variance.
    for X in [data, csr_matrix(data)]:
        X = CorrelationThreshold(threshold=.4).fit_transform(X)
        assert (len(data), 1) == X.shape


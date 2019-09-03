# Author: Lars Buitinck
# License: 3-clause BSD

import numpy as np
from ..base import BaseEstimator
from .base import SelectorMixin
from ..utils import check_array
from ..utils.sparsefuncs import mean_variance_axis, min_max_axis
from ..utils.validation import check_is_fitted
from .univariate_selection import SelectKBest, f_regression

class CorrelationThreshold(BaseEstimator, SelectorMixin):
    """Feature selector that removes all low-variance features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Read more in the :ref:`User Guide <correlation_threshold>`.

    Parameters
    ----------
    threshold : float, optional
        Features with a training-set correlation higher or equal than this threshold will
        be removed. The default is to keep all non identical features,
        i.e. remove only features that are duplicates of others.

    n_features:
        Maximum number of features to be returned.
        If not set return all features, such that the correlation is below the threshold

    Attributes
    ----------
    variances_ : array, shape (n_features,)
        Variances of individual features.

    Examples
    --------
    The following dataset has integer features, with the left most tow columns being
    linear transformations of the first. These are removed with the default setting for threshold::

        >>> X = [[2, 2, -2, 4], [0, 1, 0, 0], [1, 1, -1, 2]]
        >>> selector = CorrelationThreshold()
        >>> selector.fit_transform(X)
        array([[2, 0],
               [1, 4],
               [1, 1]])
    """

    def __init__(self, threshold=1., n_features=None):
        self.threshold = threshold
        if n_features:
            self.n_features = n_features
        else:
            self.n_features = np.inf
        self.correlations_ = None
        self.mask = None

    def fit(self, X, y=None):
        """Learn empirical variances from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute variances.

        y : {array-like, sparse matrix}, shape (n_samples, 1)
            If y given, iteratively optimize
                max Phi(S): 1/\S\    * sum_(x_i \in S) I(x_i, y)
                            - 1/\S\**2 * sum_(x_i \in S, x_j \in S) I(x_i, x_j)
            else iteratively optimize:
                max Phi(S): - 1/\S\**2 * sum_(x_i \in S, x_j \in S) I(x_i, x_j)

            with \S\ <= n_features
            and max(I(x_i, x_j)) <= threshold

        Returns
        -------
        self
        """
        X = check_array(X, dtype=float)
        n_samples, n_features = X.shape
        S = np.zeros(n_features, dtype=int)

        self.correlations_ = abs(np.corrcoef(X.transpose()))
        self.correlations_[np.isnan(self.correlations_)] = 1

        if y is not None:
            sb = SelectKBest(f_regression, "all").fit(X, y)
            I_xc = sb.pvalues_
            I_xc[np.isnan(I_xc)] = -1
        else:
            I_xc = np.zeros(n_features)

        S[np.argmax(I_xc)] = 1

        while sum(S) < self.n_features:
            # filter by threshold:
            S_mask = self.correlations_[S == 1, :].max(axis=0) < (self.threshold - 1.e-8)
            # In case no value fulfills the threshold
            if not np.any(S_mask):
                break
            else:
                # iteratively add an x_i:
                best_i = np.argmax((I_xc - 1./sum(S)*self.correlations_[S == 1, :].sum(axis=0))[S_mask])
                if S[np.where(S_mask)[0][best_i]] == 1:
                    print ("ERROR - trying to set the same value twice")
                    break
                else:
                    S[np.where(S_mask)[0][best_i]] = 1

        self.mask = S

        return self



    def _get_support_mask(self):
        check_is_fitted(self)

        return self.mask == 1

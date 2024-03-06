# -*- coding: utf-8 -*-
"""SUOD
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

try:
    import suod
except ImportError:
    print('please install suod first for SUOD by `pip install suod`')
from suod.models.base import SUOD as SUOD_model

from .base import BaseDetector
from .lof import LOF
from .hbos import HBOS
from .iforest import IForest
from .copod import COPOD
from .combination import average, maximization
from DetectOutlier.utils.utility import standardizer


class SUOD(BaseDetector):
    # noinspection PyPep8
    """SUOD (Scalable Unsupervised Outlier Detection) is an acceleration
    framework for large scale unsupervised outlier detector training and
    prediction. See :cite:`zhao2021suod` for details.

    Parameters
    ----------
    base_estimators : list, length must be greater than 1
        A list of base estimators. Certain methods must be present, e.g.,
        `fit` and `predict`.

    combination : str, optional (default='average')
        Decide how to aggregate the results from multiple models:

        - "average" : average the results from all base detectors
        - "maximization" : output the max value across all base detectors

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    n_jobs : optional (default=1)
        The number of jobs to run in parallel for both `fit` and
        `predict`. If -1, then the number of jobs is set to the
        the number of jobs that can actually run in parallel.

    rp_clf_list : list, optional (default=None)
        The list of outlier detection models to use random projection. The
        detector name should be consistent with PyOD.

    rp_ng_clf_list : list, optional (default=None)
        The list of outlier detection models NOT to use random projection. The
        detector name should be consistent with PyOD.

    rp_flag_global : bool, optional (default=True)
        If set to False, random projection is turned off for all base models.

    target_dim_frac : float in (0., 1), optional (default=0.5)
        The target compression ratio.

    jl_method : string, optional (default = 'basic')
        The JL projection method:

        - "basic": each component of the transformation matrix is taken at
          random in N(0,1).
        - "discrete", each component of the transformation matrix is taken at
          random in {-1,1}.
        - "circulant": the first row of the transformation matrix is taken at
          random in N(0,1), and each row is obtained from the previous one
          by a one-left shift.
        - "toeplitz": the first row and column of the transformation matrix
          is taken at random in N(0,1), and each diagonal has a constant value
          taken from these first vector.

    bps_flag : bool, optional (default=True)
        If set to False, balanced parallel scheduling is turned off.

    approx_clf_list : list, optional (default=None)
        The list of outlier detection models to use pseudo-supervised
        approximation. The detector name should be consistent with PyOD.

    approx_ng_clf_list : list, optional (default=None)
        The list of outlier detection models NOT to use pseudo-supervised
        approximation. The detector name should be consistent with PyOD.

    approx_flag_global : bool, optional (default=True)
        If set to False, pseudo-supervised approximation is turned off.

    approx_clf : object, optional (default: sklearn RandomForestRegressor)
        The supervised model used to approximate unsupervised models.

    cost_forecast_loc_fit : str, optional
        The location of the pretrained cost prediction forecast for training.

    cost_forecast_loc_pred : str, optional
        The location of the pretrained cost prediction forecast for prediction.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, base_estimators=None, contamination=0.1,
                 combination='average', n_jobs=None,
                 rp_clf_list=None, rp_ng_clf_list=None, rp_flag_global=True,
                 target_dim_frac=0.5, jl_method='basic', bps_flag=True,
                 approx_clf_list=None, approx_ng_clf_list=None,
                 approx_flag_global=True, approx_clf=None,
                 cost_forecast_loc_fit=None, cost_forecast_loc_pred=None,
                 verbose=False):
        super(SUOD, self).__init__(contamination=contamination)
        self.base_estimators = base_estimators
        self.contamination = contamination
        self.combination = combination
        self.n_jobs = n_jobs
        self.rp_clf_list = rp_clf_list
        self.rp_ng_clf_list = rp_ng_clf_list
        self.rp_flag_global = rp_flag_global
        self.target_dim_frac = target_dim_frac
        self.jl_method = jl_method
        self.bps_flag = bps_flag
        self.approx_clf_list = approx_clf_list
        self.approx_ng_clf_list = approx_ng_clf_list
        self.approx_flag_global = approx_flag_global
        self.approx_clf = approx_clf
        self.cost_forecast_loc_fit = cost_forecast_loc_fit
        self.cost_forecast_loc_pred = cost_forecast_loc_pred
        self.verbose = verbose

        # by default we will provide a group of performing models
        if self.base_estimators is None:
            self.base_estimators = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                                    HBOS(n_bins=10), HBOS(n_bins=20),
                                    COPOD(), IForest(n_estimators=50),
                                    IForest(n_estimators=100),
                                    IForest(n_estimators=150)]

        self.n_estimators = len(self.base_estimators)

        # pass in the arguments for SUOD model
        self.model_ = SUOD_model(
            base_estimators=self.base_estimators,
            contamination=self.contamination,
            n_jobs=self.n_jobs,
            rp_clf_list=self.rp_clf_list,
            rp_ng_clf_list=self.rp_ng_clf_list,
            rp_flag_global=self.rp_flag_global,
            target_dim_frac=self.target_dim_frac,
            jl_method=self.jl_method,
            approx_clf_list=self.approx_clf_list,
            approx_ng_clf_list=self.approx_ng_clf_list,
            approx_flag_global=self.approx_flag_global,
            approx_clf=self.approx_clf,
            bps_flag=self.bps_flag,
            cost_forecast_loc_fit=self.cost_forecast_loc_fit,
            cost_forecast_loc_pred=self.cost_forecast_loc_pred,
            verbose=self.verbose,
        )

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # validate inputs X and y (optional)
        X = check_array(X)
        n_samples, n_features = X.shape[0], X.shape[1]
        self._set_n_classes(y)

        # fit the model and then approximate it
        self.model_.fit(X)
        self.model_.approximate(X)

        # get the decision scores from each base estimators
        decision_score_mat = np.zeros([n_samples, self.n_estimators])
        for i in range(self.n_estimators):
            decision_score_mat[:, i] = self.model_.base_estimators[
                i].decision_scores_

        # the scores must be standardized before combination
        decision_score_mat, self.score_scalar_ = standardizer(
            decision_score_mat, keep_scalar=True)

        # todo: may support other combination
        if self.combination == 'average':
            decision_score = average(decision_score_mat)
        else:
            decision_score = maximization(decision_score_mat)

        assert (len(decision_score) == n_samples)

        self.decision_scores_ = decision_score.ravel()
        self._process_decision_scores()

        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detectors.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model_', 'decision_scores_',
                               'threshold_', 'labels_'])

        X = check_array(X)

        # initialize the output score
        predicted_scores = self.model_.decision_function(X)

        # standardize the score and combine
        predicted_scores = self.score_scalar_.transform(predicted_scores)

        # todo: may support other combination
        if self.combination == 'average':
            decision_score = average(predicted_scores)
        else:
            decision_score = maximization(predicted_scores)

        assert (len(decision_score) == X.shape[0])

        return decision_score.ravel()

import numpy as np

from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import invsqrtm, powm
from pyriemann.utils.distance import distance_riemann


def align_recenter_rescale(X_source, X_target):
    """Method to align source and target data by recentering and rescaling them.

    The input are MEEG covariances. First, covariances from both domains are
    re-centered to identity by whitening them with the mean covariance of each
    domain. The dispersion of the target domain is then adjusted to be equal
    to the dispersion of the source domain.

    Parameters
    ----------
    X_source: ndarray, shape (n_samples_source, n_channels, n_channels)
        Covariances of the source domain.
    X_target: ndarray, shape (n_samples_target, n_channels, n_channels)
        Covariances of the target domain.

    Returns
    ----------
    X_source_aligned: ndarray, shape (n_samples_source, n_channels, n_channels)
        Source covariances after alignment.
    X_target_aligned: ndarray, shape (n_samples_target, n_channels, n_channels)
        Target covariances after alignment.
    """
    n_source, n_channels, _ = X_source.shape
    n_target, _, _ = X_target.shape

    # Re-centering to identity
    M_source = mean_covariance(X_source)
    M_target = mean_covariance(X_target)
    M_source_inv = invsqrtm(M_source)
    M_target_inv = invsqrtm(M_target)

    X_source_rct = M_source_inv @ X_source @ M_source_inv
    X_target_rct = M_target_inv @ X_target @ M_target_inv

    # Equalizing the dispersion
    disp_source = np.sum([distance_riemann(X_source_rct[i],
                                           np.eye(n_channels)) ** 2
                          for i in range(n_source)]) / n_source
    disp_target = np.sum([distance_riemann(X_target_rct[i],
                                           np.eye(n_channels)) ** 2
                          for i in range(n_target)]) / n_target
    p_source = np.sqrt(1 / disp_source)
    p_target = np.sqrt(1 / disp_target)
    X_source_str = np.zeros(X_source.shape)
    for i in range(n_source):
        X_source_str[i] = powm(X_source_rct[i], p_source)
    X_target_str = np.zeros(X_target.shape)
    for i in range(n_target):
        X_target_str[i] = powm(X_target_rct[i], p_target)
    X_source_aligned = X_source_str
    X_target_aligned = X_target_str

    return X_source_aligned, X_target_aligned

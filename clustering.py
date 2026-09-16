from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _mclust_package():
    try:
        from rpy2.robjects.packages import importr

        return importr("mclust")
    except (ImportError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "mclust clustering requires R, the R package mclust, and rpy2. "
            "Activate the environment specified in environment.yml."
        ) from error


def check_clustering_dependencies(method="mclust"):
    if method != "mclust":
        raise ValueError("The manuscript protocol uses mclust clustering only.")
    _mclust_package()


def cluster_mclust(X, k, seed=0, return_details=False):
    matrix = np.asarray(X, dtype="float64")
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("X must contain at least two cells and one latent coordinate.")
    if not np.isfinite(matrix).all():
        raise ValueError("The clustering representation must contain only finite values.")
    if int(k) != k or not 1 <= int(k) <= matrix.shape[0]:
        raise ValueError("The reference class count must be an integer in [1, n_cells].")
    package = _mclust_package()
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(ro.default_converter + numpy2ri.converter):
        r_matrix = ro.conversion.get_conversion().py2rpy(matrix)
    ro.r["set.seed"](int(seed))
    result = package.Mclust(r_matrix, G=int(k), verbose=False)
    if result is ro.NULL or "classification" not in list(result.names):
        raise RuntimeError("mclust did not return a fitted classification.")
    labels = np.asarray(result.rx2("classification"), dtype=np.int64)
    if labels.shape != (matrix.shape[0],) or np.any((labels < 1) | (labels > int(k))):
        raise RuntimeError("mclust returned missing or invalid cell assignments.")
    components = int(result.rx2("G")[0])
    if components != int(k):
        raise RuntimeError("mclust did not fit the specified reference class count.")
    details = {
        "method": "mclust",
        "model_selection": "mclust_default_BIC",
        "covariance_model": str(result.rx2("modelName")[0]),
        "n_components": components,
    }
    if return_details:
        return labels - 1, details
    return labels - 1


def cluster_all(X, k, seed=0, methods=("mclust",)):
    unsupported = set(methods).difference({"mclust"})
    if unsupported:
        raise ValueError("The manuscript protocol uses mclust clustering only.")
    return {method: cluster_mclust(X, k, seed=seed) for method in methods}


__all__ = ["cluster_mclust", "cluster_all", "check_clustering_dependencies"]

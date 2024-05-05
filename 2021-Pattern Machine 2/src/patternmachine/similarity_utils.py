import torch
import numpy as np
from math import erf, sqrt
import functools
from config import Config
from patternmachine.utils import (
    bounded_contrast_enhancement,
    inverse,
    normalize,
    normalize_max,
    precision_to_variance,
    variance_to_precision,
)
from scipy.interpolate import interpn
import pickle


def similarity(x, x_precision, y, y_precision):
    min_precision = torch.minimum(x_precision, y_precision)
    max_precision = torch.maximum(x_precision, y_precision)

    # err = (x - y).abs()
    # sim = (1 - min_precision) * 0.2 + (1 - err) * (min_precision + max_precision) * 0.4

    # sim = sim.pow(2)

    err = (x - y).abs()
    sim = (1 - min_precision) * 0.2 + ((1 - err) * (min_precision + max_precision) * 0.5).pow(2)
    sim.clamp_(min=0, max=1)

    # shape = x.shape
    # x_abs = ((x - Config.BASE_ACTIVATION).abs() / (1 - Config.BASE_ACTIVATION)).view(shape)
    # y_abs = ((y - Config.BASE_ACTIVATION).abs() / (1 - Config.BASE_ACTIVATION)).view(shape)
    # sim = sim * x_abs * y_abs  # / (x_abs.max(dim=-1)[0] * y_abs.max(dim=-1)[0]).unsqueeze(-1)

    # a = 1.0
    # sim = a * sim2 + (1 - a) * sim
    return sim


# --- START --- Adopted from https://github.com/python/cpython/blob/5a42a49477cd601d67d81483f9589258dccb14b1/Lib/statistics.py#L970


def erf(x):
    # save the sign of x
    sign = torch.Tensor(np.where(x >= 0, 1, -1))
    x = torch.abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    z = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    y = 1.0 - z * torch.exp(-x * x)

    return sign * y


def cdf(x, mu, sigma):
    "Cumulative distribution function.  P(X <= x)"
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))


def overlap(X_mu, X_sigma, Y_mu, Y_sigma):
    """Compute the overlapping coefficient (OVL) between two normal distributions.
    Measures the agreement between two normal probability distributions.
    Returns a value between 0.0 and 1.0 giving the overlapping area in
    the two underlying probability density functions.

    See: "The overlapping coefficient as a measure of agreement between
    probability distributions and point estimation of the overlap of two
    normal densities" -- Henry F. Inman and Edwin L. Bradley Jr
    http://dx.doi.org/10.1080/03610928908830127
    """

    X_var, Y_var = X_sigma ** 2, Y_sigma ** 2

    # if not X_var or not Y_var:b
    #     raise StatisticsError('overlap() not defined when sigma is zero')
    dv = Y_var - X_var
    dm = torch.abs(Y_mu - X_mu)

    same_variance_calc = 1.0 - erf(dm / (2.0 * X_sigma * sqrt(2.0)))

    a = X_mu * Y_var - Y_mu * X_var
    b = X_sigma * Y_sigma * torch.sqrt(dm ** 2.0 + dv * torch.log(Y_var / X_var))
    x1 = (a + b) / dv
    x2 = (a - b) / dv

    diff_variance_calc = 1.0 - (
        torch.abs(cdf(x1, Y_mu, Y_sigma) - cdf(x1, X_mu, X_sigma))
        + torch.abs(cdf(x2, Y_mu, Y_sigma) - cdf(x2, X_mu, X_sigma))
    )

    return torch.Tensor(np.where(dv != 0, diff_variance_calc, same_variance_calc))


# --- END --- Adopted from https://github.com/python/cpython/blob/5a42a49477cd601d67d81483f9589258dccb14b1/Lib/statistics.py#L970

MIN_PRECISION = 0.01
MAX_PRECISION = 0.99


def _overlap_similarity(x, x_precision, y, y_precision, precision_based_selectivity=False):
    """
    Similarity:
    What is the overlap between two normal distributions represented by
    mu=x, var=inv(x_precision) and mu=y, var=inv(y_precision)?

    This distance metric has several desirable properties -
    - At the same mu values, higher precision results in lower similarity, i.e
        precision represents selectivity. This allows patterns to become increasingly
        selective and then other patterns can win for slightly different signals.
        If some regions in input space have more probability density, more patterns
        representing those regions would get learned.
    - same pixel and precision is full overlap, i.e. similarity of 1.0
    - same mu, different precision is not full overlap, less than 1.0 similarity
    - low precision generally leads to higher similarity, which allows not
        yet selective patterns to become winners. This, along with DoG initialization
        of __output__ patterns should lead to wiggling worms.
    """

    assert x.shape == x_precision.shape == y.shape == y_precision.shape
    shape = x.shape
    if len(shape) > 1:
        x = x.reshape(-1)
        x_precision = x_precision.reshape(-1)
        y = y.reshape(-1)
        y_precision = y_precision.reshape(-1)

    x_precision.clamp_(min=MIN_PRECISION, max=MAX_PRECISION)
    y_precision.clamp_(min=MIN_PRECISION, max=MAX_PRECISION)

    x_var = precision_to_variance(x_precision)
    y_var = precision_to_variance(y_precision)
    sim_1 = overlap(
        x,
        x_var,
        y,
        y_var,
    )

    if precision_based_selectivity:
        sim_1 = sim_1 * normalize_max(1.0 / x_precision) * normalize_max(1.0 / y_precision)
        #     # sp = sim_1 / (x_precision * y_precision)
        #     # sp = sim_1 * (1 - x_precision) * (1 - y_precision)
        #     # spf = 0.9
        #     # sim_1 = sim_1 * (1 - spf) + sp * spf
        #     # sim_1.clamp_(min=0.01, max=0.99)
        #     sim_1 *= x_precision * y_precision
        # sim_1 *= inverse(x_precision * y_precision)

    # sim_1 = sim_1 * (x_precision.max() / x_precision) * (y_precision.max() / y_precision)
    if len(shape) > 1:
        sim_1 = sim_1.view(shape)
        # * (
        #     1.0
        #     - (
        #         x_precision.view(shape).mean(dim=-1).unsqueeze(-1)
        #         * y_precision.view(shape).mean(dim=-1).unsqueeze(-1)
        #     )
        # )

        # sim_1.clamp_(min=0, max=1)

    # pixels and patterns close to base activation (0.3) are not significant
    # also, divide by max so that contrast is normalized to 1
    # COMMENTED OUT: if x and pattern are highly precise at pixel
    # value 0.3, that is ok. Say a pattern that is all 0.3 pixels
    # matches highly with input patch that is all 0.3 pixels. As long as
    # sufficient number of winners are chosen, such patterns winning
    # should not be of concern. Maybe they are wasteful and prevent
    # more information-rich patterns from winning? Hopefully not.
    # x_abs = ((x - Config.BASE_ACTIVATION).abs() / (1 - Config.BASE_ACTIVATION)).view(shape)
    # y_abs = ((y - Config.BASE_ACTIVATION).abs() / (1 - Config.BASE_ACTIVATION)).view(shape)
    # sim_1 = sim_1 * x_abs * y_abs  # / (x_abs.max(dim=-1)[0] * y_abs.max(dim=-1)[0]).unsqueeze(-1)

    # higher precision is preferred
    # sim_1 = sim_1 * x_precision.view(shape) * y_precision.view(shape)

    # also, higher max precision indicates mature patterns, which are more selective
    # sim_1 = (
    #     sim_1
    #     * (1 - x_precision.view(shape).max(dim=-1)[0].unsqueeze(-1))
    #     * (1 - y_precision.view(shape).max(dim=-1)[0].unsqueeze(-1))
    # )

    # sim_1 = sim_1 * (1 - x_precision.view(shape)) * (1 - y_precision.view(shape))

    return sim_1


N = 50
M = 100
memo = None


def memo_file_name(name, n, m):
    return f"{name}_{n}_{m}.pkl"


def load_memo(name, n, m):
    try:
        with open(memo_file_name(name, n, m), "rb") as f:
            return pickle.load(f)
    except:
        return None


def save_memo(name, n, m, obj):
    with open(memo_file_name(name, n, m), "wb") as f:
        pickle.dump(obj, f)


def overlap_similarity_exp_memoized(
    x, x_precision, y, y_precision, precision_based_selectivity=True
):
    global memo
    if memo is None:
        memo = load_memo("overlap_similarity_exp_memoized", N, M)

    if memo is None:
        x1, p1, x2, p2 = torch.meshgrid(
            torch.linspace(0.0, 1.0, steps=N),
            torch.linspace(np.exp(MIN_PRECISION).item(), np.exp(MAX_PRECISION).item(), steps=M),
            torch.linspace(0.0, 1.0, steps=N),
            torch.linspace(np.exp(MIN_PRECISION).item(), np.exp(MAX_PRECISION).item(), steps=M),
        )

        memo = _overlap_similarity(x1, np.log(p1), x2, np.log(p2))
        save_memo("overlap_similarity_exp_memoized", N, M, memo)

    pi1 = (np.exp(x_precision) - np.exp(MIN_PRECISION)) / np.exp(MAX_PRECISION)
    pi2 = (np.exp(y_precision) - np.exp(MIN_PRECISION)) / np.exp(MAX_PRECISION)

    i1 = (x * (N - 1)).long()
    i2 = (x_precision * (M - 1)).long()
    i3 = (y * (N - 1)).long()
    i4 = (y_precision * (M - 1)).long()

    return memo[i1, i2, i3, i4]


lin_memo = None


def overlap_similarity_lin_memoized(
    x, x_precision, y, y_precision, precision_based_selectivity=True
):
    global lin_memo
    if lin_memo is None:
        lin_memo = load_memo("overlap_similarity_lin_memoized", N, M)

    if lin_memo is None:
        x1, p1, x2, p2 = torch.meshgrid(
            torch.linspace(0.0, 1.0, steps=N),
            torch.linspace(MIN_PRECISION, MAX_PRECISION, steps=M),
            torch.linspace(0.0, 1.0, steps=N),
            torch.linspace(MIN_PRECISION, MAX_PRECISION, steps=M),
        )

        lin_memo = _overlap_similarity(x1, p1, x2, p2)
        save_memo("overlap_similarity_lin_memoized", N, M, lin_memo)

    i1 = (x * (N - 1)).long()
    i2 = (x_precision * (M - 1)).long()
    i3 = (y * (N - 1)).long()
    i4 = (y_precision * (M - 1)).long()

    return lin_memo[i1, i2, i3, i4]


lin_inter_memo_points = None
lin_inter_memo_values = None


def overlap_similarity_lin_interpolate(
    x, x_precision, y, y_precision, precision_based_selectivity=True
):
    global lin_inter_memo_points, lin_inter_memo_values
    if lin_inter_memo_values is None:
        lin_inter_memo_values = load_memo("lin_inter_memo_values", N, M)
        lin_inter_memo_points = load_memo("lin_inter_memo_points", N, M)

    if lin_inter_memo_values is None or lin_inter_memo_points is None:
        lin_inter_memo_points = (
            torch.linspace(0.0, 1.0, steps=N),
            torch.linspace(MIN_PRECISION, MAX_PRECISION, steps=M),
            torch.linspace(0.0, 1.0, steps=N),
            torch.linspace(MIN_PRECISION, MAX_PRECISION, steps=M),
        )
        x1, p1, x2, p2 = torch.meshgrid(*lin_inter_memo_points)

        lin_inter_memo_values = _overlap_similarity(x1, p1, x2, p2)

        save_memo("lin_inter_memo_values", N, M, lin_inter_memo_values)
        save_memo("lin_inter_memo_points", N, M, lin_inter_memo_points)

    return interpn(
        [x.numpy() for x in lin_inter_memo_points],
        lin_inter_memo_values.numpy(),
        torch.stack([x, x_precision, y, y_precision]).permute(1, 2, 0).numpy(),
    )


overlap_similarity = overlap_similarity_lin_memoized
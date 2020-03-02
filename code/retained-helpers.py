# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
# -

# Colors used for plotting the posterior predictives
COLORS = {
    "true": "tab:orange",
    "predicted": "tab:blue",
    "calibrated": "tab:pink",
    "observations": "lightgrey",
}
# Transparency for the posterior predictives
FILL_ALPHA = 0.1


def generate_data(func, points, seed=0):
    """Generate a dataframe containing the covariate X, and observations Y

    The X's are generated uniformly over each of the supplied segments.

    Args:
        func: a scipy.stats function
        points: a list of dictionaries describing the points
            The expected format: [{"n_points": 10, "xlim": [-1, 1]}, ...]
        seed: random seed (default: {0})

    Returns:
        a pandas DataFrame with the generated X and Y
    """
    np.random.seed(seed)

    data = []
    for segment in points:
        x = np.linspace(*segment["xlim"], num=segment["n_points"])
        distribution = func(x)
        # Generate observations
        y = distribution.rvs()
        df = pd.DataFrame({"x": x, "y": y})
        data.append(df)

    return pd.concat(data, ignore_index=True)


def retain(func, df, frac, seed=None):
    """Retain a fraction of the original data corresponding to the lowest
    posterior predictive uncertainty.
    
    Args:
        func: a scipy.stats distribution of the posterior predictive
        df: a pandas DataFrame with the data
        frac: a fraction of the data to retain
        seed: an optional random seed used to break the ties
    """
    # Randomize the order of rows for breaking the ties
    df = df.copy().sample(frac=1, random_state=seed)

    # Retrieve the uncertainty estimates of the posterior predictive for each X
    dist = func(df.x)
    df["_uncertainty"] = dist.std()

    # Retain a portion of the observations with the lowest uncertainty
    n = int(df.shape[0] * frac)
    df_retained = df.nsmallest(n, "_uncertainty").sort_index().drop(columns="_uncertainty")

    return df_retained


# +
def rmse(y_true, y_pred):
    """Compute RMSE"""
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def mae(y_true, y_pred):
    """Compute MAE"""
    return (y_pred - y_true).mean()


# -

class FakePosterior:
    """Fake posterior predictive centered around zero with a polynomial mean and homoscedastic aleatoric noise.
    
    Args:
        x: the predictor variable to fit to
        y: the response variable
        degree: a polynomial degree for the fitted mean
        gap: a tuple with the range of the gap for epistemic uncertainty
        aleatoric: standard deviation of aleatoric noise (outside the gap)
        epistemic: the maximum standard deviation for epistemic uncertainty (in the gap region)
        center: the point at which epistemic uncertainty is the largest (default: 0)
    """
    def __init__(self, x, y, degree, aleatoric, epistemic, gap, center=0):
        self._poly = np.poly1d(np.polyfit(x, y, deg=degree))
        low, high = gap
        xx = np.concatenate((np.linspace(x.min(), low), [center], np.linspace(high, x.max())))
        yy = np.concatenate((np.full(50, aleatoric), [aleatoric + epistemic], np.full(50, aleatoric)))
        self._spline = UnivariateSpline(xx, yy, k=3, s=0)
        
    def __call__(self, x):
        self._x = x
        return self
        
    def mean(self):
        return self._poly(self._x)
    
    def std(self):
        return self._spline(self._x)
        
    def interval(self, interval):
        assert 0 <= interval <= 1
        # Assuming symmetric Gaussian noise
        q_alpha = 1 - interval
        z = scipy.stats.norm.ppf(interval + q_alpha / 2)
        mean = self.mean()
        margin_error = z * self.std()
        return mean - margin_error, mean + margin_error


def plot(func, df, name, interval=0.95, observations=True, title=None, legend=True, ax=None):
    """Plot the distribution function and the observations

    Args:
        func: a scipy.stats distribution
        df: a pandas DataFrame containing observations (x, y)
        name: a description of the distribution function, e.g. "true" or "predicted"
        interval: the width of the predictive interval (default: 0.95)
        observations: optionally plot the observations (default: True)
        title: an optional plot title (default: None)
        legend: whether to show a legend (default: True)
        ax: matplotlib axis to draw on, if any (default: None)
    """
    assert 0 <= interval <= 1

    x = np.linspace(df.x.min(), df.x.max(), num=1000)
    distribution = func(x)
    lower, upper = distribution.interval(interval)
    point_est = distribution.mean()

    ax = ax or plt.gca()
    ax.fill_between(
        x,
        lower,
        upper,
        color=COLORS[name],
        alpha=FILL_ALPHA,
        label=f"{name.title()} {interval*100:.0f}% Interval",
    )
    if observations:
        ax.scatter(df.x, df.y, s=10, color=COLORS["observations"], label="Observations")
    ax.plot(x, point_est, color=COLORS[name], label=f"{name.title()} Mean")
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)


# +
def plot_true_function(*args, **kwargs):
    plot(*args, **kwargs, name="true")


def plot_posterior_predictive(*args, **kwargs):
    plot(*args, **kwargs, name="predicted", observations=False)


# -

def plot_rmse(true_func, ppc_func, df, fractions=None, label=None, seed=0, title=None):
    """Visualize RMSE for the multiple fractions of retained data.
    
    Args:
        true_func: a scipy.stats distribution corresponding to the true function
        ppc_func: a scipy.stats distribution for the posterior predictive
        df: a pandas DataFrame with the predictor variable X
        fractions: an optional list of fractions of retained data (default: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        label: an optional legend label for the curve
        seed: an optional random seed to break the ties when retaining data (default: 0)
    """
    if fractions is None:
        fractions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    metrics = {}
    for frac in fractions:
        df_retained = retain(ppc_func, df, frac=frac, seed=seed)

        y_true = df_retained.y.values
        y_pred = ppc_func(df_retained.x).mean()
        metrics[frac] = rmse(y_true, y_pred)

    pd.Series(metrics).plot(style="-o", xlim=[min(fractions) - 0.02, 1.02], label=label.title())
    plt.xlabel("Fraction of Retained Data")
    plt.ylabel("RMSE")
    plt.title(title or "RMSE vs Fraction of Retained Data")
    plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0, title="Posterior Predictives")

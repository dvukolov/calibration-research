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
import scipy.stats

import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
# %run retained-helpers.ipynb
# -

# %config InlineBackend.figure_format='retina'
plt.rc("figure", figsize=(6, 4), dpi=100)
sns.set(style="white", palette=None, rc={"axes.linewidth": 1})


def heteroscedastic_base(x, base_std, bias=0):
    """A heteroscedastic function that serves as the basis for
    the true function and a hypothetical posterior predictive.
    
    Args:
        x: a 1D array of predictor values
        base_std: a scalar used for computing the standard deviation of Y
        bias: a scalar term introducing bias to the mean prediction
    """
    std = np.abs(x) * base_std
    std = np.where(std < 0.5, 0.5, std)
    return scipy.stats.norm(loc=(0.1 + bias) * x ** 3, scale=std)


# # Original plots for classification using Accuracy and AUC

# ![gals-retained-metric.png](attachment:gals-retained-metric.png)

# # Similar plots for regression based e.g. on RMSE
#
# ## Case 1: The role of aleatoric uncertainty and bias

# +
# Define the true function and generate observations
heteroscedastic = lambda x: heteroscedastic_base(x, base_std=1.0)

data_points = [
    {"n_points": 200, "xlim": [-4, 4]},
]
df = generate_data(heteroscedastic, points=data_points, seed=4)

# Plot the data
plot_true_function(heteroscedastic, df, title=fr"True Function: $y_i = 0.1x_i^3 + \varepsilon_i$")

# +
ppc = {}

ppc["any unbiased"] = lambda x: scipy.stats.norm(loc=0.10 * x ** 3, scale=1)
plot_true_function(heteroscedastic, df)
plot_posterior_predictive(ppc["any unbiased"], df, title="Any Posterior Predictive with Unbiased Mean")
# -

ppc["homoscedastic biased"] = lambda x: scipy.stats.norm(loc=0.15 * x ** 3, scale=1)
plot_true_function(heteroscedastic, df)
plot_posterior_predictive(ppc["homoscedastic biased"], df, title="Homoscedastic Biased Posterior Predictive")

# +
ppc["heteroscedastic narrow"] = lambda x: heteroscedastic_base(x, base_std=0.5, bias=0.05)

plot_true_function(heteroscedastic, df)
plot_posterior_predictive(
    ppc["heteroscedastic narrow"], df, title="Heteroscedastic Biased Posterior Predictive"
)
# -

plot_true_function(heteroscedastic, df)
plot_posterior_predictive(
    ppc["heteroscedastic narrow"], df, title="Possible to Defer The Most Uncertain Predictions"
)
plt.axvspan(-4, -3, ymin=0.01, ymax=0.99, alpha=0.7, color="white", zorder=3)
plt.axvspan(3, 4, ymin=0.01, ymax=0.99, alpha=0.7, color="white", zorder=3);

# +
ppc["heteroscedastic wide"] = lambda x: heteroscedastic_base(x, base_std=1.0, bias=0.049)

plot_true_function(heteroscedastic, df)
plot_posterior_predictive(
    ppc["heteroscedastic wide"], df, title="Heteroscedastic Posterior Predictive with More Uncertainty"
)

# +
ppc["heteroscedastic less biased"] = lambda x: heteroscedastic_base(x, base_std=0.5, bias=0.02)

plot_true_function(heteroscedastic, df)
plot_posterior_predictive(
    ppc["heteroscedastic less biased"], df, title="Heteroscedastic (Less Biased) Posterior Predictive"
)
# -

models = [
    "heteroscedastic wide",
    "heteroscedastic narrow",
    "heteroscedastic less biased",
    "homoscedastic biased",
    "any unbiased",
]
for model in models:
    plot_rmse(ppc_func=ppc[model], df=df, label=model)

# - Models with proper heteroscedastic aleatoric (or total) uncertainty can improve their performance by deferring predictions for the most uncertain cases
# - The slope of the line indicates the degree of this improvement, i.e. steeper slopes are making better use the of uncertainty, all else held equal (as noted by Gal et al.)
# - Models with homoscedastic aleatoric (or total) uncertainty do not have this property
# - Less biased and more accurate models lie to the lower right on the RMSE plot
# - Performance could be measured by the area under the curve, which we would seek to minimize
# - Heteroscedastic models with proportionally higher or lower uncertainties are not distinguished by the plot. Absolute values of uncertainty are not taken into account.
# - The original plots additionally visualize the uncertainty of the metric itself (e.g. accuracy $\pm 1 \text{ std.}$)
#     - How does one compute the confidence intervals for RMSE or other regression metrics e.g. NLL?
#     - Do they depend on aleatoric uncertainty only, i.e. the variance of the mean prediction or on both aleatoric & epistemic?
# - We could construct the same plot with other metrics instead of RMSE, e.g. the negative log-likelihood
# - (Optionally) A [normalized RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalized_root-mean-square_deviation) may be used to facilitate comparisons between datasets or models with different scales

# ![gals-retained-metric.png](attachment:gals-retained-metric.png)

# ## Case 2: Good aleatoric, varying epistemic uncertainty 

# +
# Define the true function and generate observations
homoscedastic = lambda x: scipy.stats.norm(loc=0.1 * x ** 3, scale=1.0)

train_points = [
    {"n_points": 40, "xlim": [-4, -1.5]},
    {"n_points": 40, "xlim": [1.5, 4]},
]
test_points = [
    {"n_points": 120, "xlim": [-4, 4]},
]
df = generate_data(homoscedastic, points=train_points, seed=0)
df_test = generate_data(homoscedastic, points=test_points, seed=1)

# Plot the data
plot_true_function(homoscedastic, df, title=fr"True Function: $y_i = 0.1x_i^3 + \varepsilon$")

# +
ppc["adequate uncertainty"] = FakePosterior(
    df.x, df.y, degree=8, aleatoric=1.0, epistemic=2.0, gap=[-1.7, 2.0]
)

plot_true_function(homoscedastic, df)
plot_posterior_predictive(ppc["adequate uncertainty"], df, title="Adequate Posterior Predictive")

# +
ppc["large epistemic"] = FakePosterior(df.x, df.y, degree=10, aleatoric=1.0, epistemic=5.0, gap=[-1.7, 2.0])

plot_true_function(homoscedastic, df)
plot_posterior_predictive(ppc["large epistemic"], df, title="Large Epistemic Uncertainty")

# +
ppc["small epistemic"] = FakePosterior(df.x, df.y, degree=6, aleatoric=1.0, epistemic=0.2, gap=[-1.7, 2.0])

plot_true_function(homoscedastic, df)
plot_posterior_predictive(ppc["small epistemic"], df, title="Small Epistemic Uncertainty")

# +
ppc["even smaller epistemic"] = FakePosterior(
    df.x, df.y, degree=4, aleatoric=1.0, epistemic=0.2, gap=[-1.7, 2.0]
)

plot_true_function(homoscedastic, df)
plot_posterior_predictive(ppc["even smaller epistemic"], df, title="Even Smaller Epistemic Uncertainty")

# +
models = ["adequate uncertainty", "large epistemic", "small epistemic", "even smaller epistemic"]

for model in models:
    plot_rmse(
        ppc_func=ppc[model], df=df, label=model, title="RMSE vs Retained Fraction on Training Data",
    )
# -

# - The effect of data retention is random
# - On the training set or with no out-of-distribution data, epistemic uncertainty plays a very minor role, if any.
# - Admittedly, this example is extreme, and in real-world scenarios some non-random effects may be expected

# +
models = ["adequate uncertainty", "large epistemic", "small epistemic", "even smaller epistemic"]

for model in models:
    plot_rmse(
        true_func=homoscedastic,
        ppc_func=ppc[model],
        df=df_test,
        label=model,
        title="RMSE vs Retained Fraction on Testing Data",
    )
# -

# - On the test set, *with the need to predict out-of-distribution data*, the effects of epistemic uncertainty can be observed
# - To be precise, the plot depicts the deviation of the mean prediction from the true values
# - The mean prediction, in turn, has a tendency to vary depending on the amount of epistemic uncertainty
# - Thus the presence of epistemic uncertainty allows to improve model performance when a part of the data is retained
# - The absolute amount of epistemic uncertainty matters as long as it reflects the variability of the mean prediction
# - Otherwise, the exact absolute amount of epistemic uncertainty plays no role
# - We *need to have out-of-distribution data* to be able to distinguish a Bayesian model with good epistemic uncertainty from a model with bad epistemic uncertainty one with this plot.

# # Next steps

# - Further investigation into these plots and metrics
#     - Construct additional toy examples of good and bad Bayesian models to understand if the plot can distinguish between them
#     - Apply the calibration algorithm and see if the potentially positive effects of calibration are reflected in the plots
#     - Try other metrics in addition to RMSE, such as negative log-likelihood, etc.
#     - Understand how confidence intervals for each metric could be calculated
#     - Produce similar diagnostic plots for more realistic datasets, e.g. UCI datasets
# - Calibration
#     - Understand how we could apply the fairness criterion for informed (not average) calibration
#     - Research possible ways to calibrate without ruining epistemic uncertainty
#     - See if it is possible to automatically detect where (in which regiond or groupd) the model is miscalibrated
# - Application
#     - Pick a couple of candidate tasks in active learning, RL or Bayesian optimization
#     - Come up with ways to more comprehensively evaluate models with uncertainty on those tasks



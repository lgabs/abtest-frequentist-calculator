import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.proportion import proportions_ztest
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

plt.style.use("fivethirtyeight")

import plotly.express as px

import streamlit as slit

import logging

# logging.basicConfig(level=logging.INFO)

# matplotlib.rcParams['text.usetex'] = True
# colors for prints
CRED = "\033[91m"
CGREEN = "\33[32m"
CEND = "\033[0m"


def generate_binomial_data(
    sizeA=100,
    sizeB=100,
    pA=0.2,
    pB=0.7,
):
    """
    Generate random data (like from ads in pages) for simulating purposes from binomial distribution.
    """

    dfA = pd.DataFrame(
        data={"variant": ["A"] * sizeA, "target": np.random.binomial(1, pA, sizeA)}
    )
    dfB = pd.DataFrame(
        data={"variant": ["B"] * sizeB, "target": np.random.binomial(1, pB, sizeB)}
    )
    df = dfA.append(dfB).sample(frac=1).reset_index(drop=True)

    return df


def generate_conversion_data(sizes=[], convs=[], labels=[]):
    """
    Generate data with exact convertions for simulating purposes.
    """
    df = pd.DataFrame(columns=["variant", "target"])
    for conv, size, label in zip(convs, sizes, labels):
        df_conv = pd.DataFrame(data={"variant": [label] * conv, "target": [1] * conv})
        df_notconv = pd.DataFrame(
            data={"variant": [label] * (size - conv), "target": [0] * (size - conv)}
        )
        df = df.append(df_conv).append(df_notconv)

    df = df.sample(frac=1).reset_index(drop=True)

    return df


def confidence_interval(X, significance=0.05, sigma_estimation=None):
    """
    t-confidence interval.
    Args:
        X (np.array): data
        significance (float): significance of CI (confidence=1-significance)
    Returns:
        A tuple with mu hat and CI edges
    """

    N = len(X)
    mu_hat = np.mean(X)
    if sigma_estimation is None:
        sigma_estimation = np.std(X, ddof=1)
    z_left = st.norm.ppf(significance / 2)
    z_right = st.norm.ppf((1 - significance / 2))
    lower = mu_hat + z_left * sigma_estimation / np.sqrt(N)
    upper = mu_hat + z_right * sigma_estimation / np.sqrt(N)

    return (mu_hat, lower, upper)


def estimate_sample_size(
    significance=0.05,
    power=0.80,
    min_diff=None,
    effect_type="absolute",
    mu_baseline=None,
    objective_metric_type="binary",
    test_type="one-sided",
    sigma_estimation=None,
    estimated_impressions_daily=None,
    streamlit_print=False,
    verbose=True,
):
    """
    Args:
        significance (float): desired significance for the test.
        power (float): desired power of the test.
        min_diff (float): desired minimum detectable effect in objective metric (relative or absolute in next param)
        effect_type: 'absolute' or 'relative'
        mu_baseline (float): baseline mean for objetivce metric (e.g. conversion rate, revenue).
                             If metric is binary like conversion rate, mu_baseline is the conversion rate or CTR.
        objective_metric_type (str): type of objective metric. If 'binary' and 'sigma_estimation' is None, sigma will be estimated with np.sqrt(p(1-p)) where p=mu.
        sigma_estimation (float): estimation of sigma, if user wants to use some pre-defined value
        estimated_impressions_daily (int): Impressions per variation daily.
                                           If given, estimates how many days and weeks would it be necessary to run the desired A/B test with given infos.
    Returns:
        Sample size (int)
    """

    if effect_type == "relative":
        # convert to absolute
        min_diff = min_diff * mu_baseline

    logging.info(f"effect_type: {effect_type}")
    logging.info(f"Test type: {test_type}")

    beta = 1 - power
    if test_type == "two-sided":
        z_alpha = st.norm.ppf(1 - significance / 2)
    elif test_type == "one-sided":
        z_alpha = st.norm.ppf(1 - significance)
    else:
        raise ValueError("You should define between one-sided and two-sided.")

    z_beta = st.norm.ppf(1 - beta)

    if objective_metric_type == "binary":
        if verbose:
            logging.info(f"objective metric type: {objective_metric_type}")
        if sigma_estimation is None:
            p = mu_baseline
            sd1 = np.sqrt(2 * p * (1.0 - p))
            sd2 = np.sqrt(p * (1.0 - p) + (p + min_diff) * (1.0 - p - min_diff))
        else:
            sd1 = sigma_estimation
            sd2 = sigma_estimation
    elif objective_metric_type == 'continuous':
        # TODO: code this part
        raise NotImplementedError('This type of objective metric is not implemented.')
    else:
        raise ValueError("objective_metric_type must be either 'binary' or 'continuous'")


    # estimate sample size n
    n = int(round((z_alpha * sd1 + z_beta * sd2) ** 2 / min_diff**2))

    if verbose:
        print(f"baseline mean: {round(100*mu_baseline,2)}%")
        print(f"min diff absoute: {round(100*min_diff,2)}%")
        print(f"min diff relative: {round(100*min_diff/mu_baseline,2)}%")
        print("sigma1, sigma2: ", sd1, sd2)
        print(f"min_diff is {round(min_diff / sd1, 2)} sigma1's")
        print(f"estimate for sample size: {n} samples per variation.")

    phrase_days_estimations = ""
    if estimated_impressions_daily and verbose:
        days = np.ceil(n / estimated_impressions_daily)
        weeks = round(days / 7, 2)
        phrase_days_estimations = f"With {estimated_impressions_daily} impressions per day, you will need about {days} days or {weeks} weeks to run this A/B Test."
        print(phrase_days_estimations)

    if streamlit_print:

        slit.write(
            f"""
        effect_type: {effect_type}\n
        Test type: {test_type}\n
        objective metric type: {objective_metric_type}

        baseline mean: {round(100*mu_baseline,2)}%\n
        min diff absoute: {round(100*min_diff,2)}%\n
        min diff relative: {round(100*min_diff/mu_baseline,2)}%\n
        sigma1, sigma2: {sd1}, {sd2}\n
        min_diff is {round(min_diff / sd1, 2)} sigma1's

        ### Estimate for sample size: {n} samples per variation.

        {phrase_days_estimations}
        """
        )

    return n


def plot_sample_sizes(
    min_diff_range,
    mu_baseline=None,
    significance=0.05,
    power=0.80,
    verbose=False,
    streamlit_plot=False,
    **kargs,
):
    sample_sizes = [] * len(min_diff_range)
    for min_diff in min_diff_range:
        sample_sizes.append(
            estimate_sample_size(
                min_diff=min_diff,
                verbose=verbose,
                mu_baseline=mu_baseline,
                significance=significance,
                power=power,
                **kargs,
            )
        )

    f, ax = plt.subplots(figsize=(9, 7))
    df = pd.DataFrame(
        data={
            "Minimum Difference in Means": min_diff_range,
            "Sample Size Required": sample_sizes,
        }
    )
    fig = px.line(
        df,
        x="Minimum Difference in Means",
        y="Sample Size Required",
        template="seaborn",
    )
    fig.update_layout(
        title=f"Sample sizes Estimations <br> alpha={significance}, power={power}, mean_baseline: {mu_baseline}",
        xaxis_title="Expected Minimum Difference in Means (absolute values)",
        yaxis_title="Sample Size Required per Variantion",
        font=dict(family="Courier New, monospace", size=10, color="RebeccaPurple"),
    )

    if streamlit_plot:
        slit.plotly_chart(fig, use_container_width=False)


def estimate_minimum_detectable_diff(
    significance=0.05,
    power=0.80,
    current_sample_size=None,
    mu_baseline=None,
    objective_metric_type="binary",
    test_type="one-sided",
    sigma_estimation=None,
):

    beta = 1 - power
    if test_type == "two-sided":
        z_alpha = st.norm.ppf(1 - significance / 2)
    elif test_type == "one-sided":
        z_alpha = st.norm.ppf(1 - significance)
    else:
        raise ValueError("You should define between one-sided and two-sided.")

    z_beta = st.norm.ppf(1 - beta)

    if objective_metric_type == "binary":
        if sigma_estimation is None:
            p = mu_baseline
            sd1 = np.sqrt(2 * p * (1.0 - p))
            sd2 = sd1
        else:
            sd1 = sigma_estimation
            sd2 = sigma_estimation

    # estimate detectable diff
    dect_diff = (z_alpha * sd1 + z_beta * sd2) / np.sqrt(current_sample_size)

    return dect_diff


class FrequentistExperiment:

    """
    Frequentist experiment for two variants 'A' (baseline) and 'B' (which we want to test).
    """

    def __init__(
        self,
        df,
        diff_baseline=0,
        test_dist="proportions-ztest",
        alternative="larger",
        significance=0.05,
        power=0.80,
        labels=["A", "B"],
        verbose=True,
    ):
        """
        Args:
            df (pd.DataFrame): dataframe with data. Must have two columns: 'variant' (str) with variant names and 'target' (float) with observed values.
            diff_baseline (float): baseline difference between means
            test_dist (str): distribution for target values. 'proportions-ztest' is default for binary and 'ztest' for continuous means
            alternative (str): type of test. 'larger' or right-sided, like statsmodel definitions
            significance (float): significance for the test
        """

        self.df = df
        self.labels = labels
        x1 = df.query(f"variant=='{labels[0]}'")["target"].dropna().to_numpy()
        x2 = df.query(f"variant=='{labels[1]}'")["target"].dropna().to_numpy()
        self.x1 = x1
        if len(x2) == 0:
            self.x2 = None
        else:
            self.x2 = x2

        self.diff_baseline = diff_baseline
        self.test_dist = test_dist
        self.alternative = alternative
        self.significance = significance
        self.power = power
        self.sample_size = None  # minimum sample size
        self.verbose = verbose

        if alternative == "larger" and self.verbose:
            logging.warning(
                "Default alternative='larger' param stand for checking whether x2_metric-x1_metric is relevant, i.e., it's a right-sided test."
            )

    def update_mu_baseline(self, mu_baseline):
        self.mu_baseline = mu_baseline

    def update_minimum_sample_size(self, n):
        """
        Update minimum sample size for later comparison when running an A/B test, so user can know whether the test reached desired sample size.
        """
        self.sample_size = n

    def run(self):
        if self.test_dist == "ztest":
            z, p = ztest(
                x1=self.x2,
                x2=self.x1,
                value=self.diff_baseline,
                alternative=self.alternative,
            )

        elif self.test_dist == "proportions-ztest":
            self.objective_metric_type = "binary"
            sucess1 = np.count_nonzero(self.x1 == 1)
            sucess2 = np.count_nonzero(self.x2 == 1)
            # order is inverse because we want to test if p2 - p1 > diff_baseline and lib tests p1 - p2 > diff_baseline
            # https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html
            successes = np.array([sucess2, sucess1])
            samples = np.array([len(self.x2), len(self.x1)])
            z, p = proportions_ztest(
                count=successes, nobs=samples, alternative=self.alternative
            )

        if self.verbose:
            print("-" * 30 + "\nExperiment finished.\n" + "-" * 30)
            print(f"samples in x1: ", len(self.x1))
            print(f"samples in x2: ", len(self.x2))
            print(
                f"z statistic: {z}\np-value: {p}\nsignificance: {self.significance}\ndiff in means under H0: {self.diff_baseline}"
            )
            print(
                f"Means/Proportions:\n\tx1: {round(self.x1.mean(),4)}\n\tx2: {round(self.x2.mean(),4)}"
            )
            print(
                f"\tx2 - x1: {round(self.x2.mean() - self.x1.mean(),4)} ({round(100. * (self.x2.mean() / self.x1.mean() - 1), 2 ) }%)"
            )

            if self.alternative == "larger":
                self.test_type = "one-sided"
            else:
                self.test_type = "two-sided"

            # checks if minimum sample size was defined
            if not self.sample_size:
                logging.warning(
                    "You did not set previously any Minimum Sample Size to your Experiment.\n"
                    + CRED
                    + "Do not trust"
                    + CEND
                    + " on test's statistic without defining your sample size before and waiting until your data reaches it."
                )
            else:
                self.detectable_size = estimate_minimum_detectable_diff(
                    self.significance,
                    self.power,
                    len(self.x2),
                    self.mu_baseline,
                    self.objective_metric_type,
                    self.test_type,
                )
                print(
                    f"\nFor your sample size, the detectable difference estimated is: {self.detectable_size}"
                )
                if min(len(self.x1), len(self.x2)) < self.sample_size:
                    logging.warning(
                        f"Your experiment"
                        + CRED
                        + " DID NOT REACH "
                        + CEND
                        + f"minimum {self.sample_size} sample size for one or more variants"
                    )
                else:
                    print(
                        f"\nYou experiment "
                        + CGREEN
                        + "HAS SUFFICIENT IMPRESSIONS "
                        + CEND
                        + "for both variants compared to minimum {self.sample_size} sample size."
                    )

            if p <= self.significance:
                print(
                    "You "
                    + CGREEN
                    + "CAN REJECT"
                    + CEND
                    + f" the null hypothesis for significance {self.significance}."
                )
            else:
                print(
                    "You "
                    + CRED
                    + "CAN NOT REJECT"
                    + CEND
                    + f" the null hypothesis for significance {self.significance}."
                )

        return {"statistic": z, "pvalue": p, "passed": (p <= self.significance)}

    #     def plot_data_distribution(self, x1label="x1", x2label="x2", kind="kde"):

    #         """
    #         Plot histograms of data or KDE approximations.
    #         """

    #         plt.rc('legend', fontsize=15)
    #         f, ax = plt.subplots(figsize=(9,7))

    #         if kind == "kde":
    #             print("plotting distributions using seaborn's KDE method.")
    #             sns.kdeplot(self.x1, label=f'{x1label} ($\mu={round(self.x1.mean(),2)})$', ax=ax)
    #             sns.kdeplot(self.x2, label=f'{x2label} ($\mu={round(self.x2.mean(),2)})$', ax=ax)
    #         elif kind == "hist":
    #             sns.histplot(self.x1, label=f'{x1label} ($\mu={round(self.x1.mean(),2)})$', ax=ax)
    #             sns.histplot(self.x2, label=f'{x2label} ($\mu={round(self.x2.mean(),2)})$', ax=ax)

    #         plt.legend();

    def plot_data_distribution(self):

        f, ax = plt.subplots(figsize=(9, 7))

        plot_norm_distribution(self.x1, self.test_dist, ax=ax, variant="A")
        plot_norm_distribution(self.x2, self.test_dist, ax=ax, variant="B")

        title = f"Normal Distributions for Variants"
        plt.title(title)

        plt.legend()

    def plot_timeline(self, step=100, ax=None, **kargs):
        plot_timeline_experiments(
            df=self.df,
            labels=self.labels,
            step=step,
            ax=ax,
            sample_size=self.sample_size,
            **kargs,
        )


def plot_norm_distribution(x, test_dist="proportions-ztest", ax=None, variant=None):

    if not ax:
        f, ax = plt.subplots(figsize=(9, 7))

    # samples
    N = len(x)
    # get mean
    mu_hat = x.mean()
    # get std
    if test_dist == "ztest":
        sigma_hat = x.std(ddof=1)  # ddof=1 tu use unbiased calc
    elif test_dist == "proportions-ztest":
        p = mu_hat
        sigma_hat = np.sqrt(p * (1 - p))

    samples = st.norm.rvs(loc=mu_hat, scale=sigma_hat / np.sqrt(N), size=1000)
    xmin, xmax = np.min(samples), np.max(samples)

    x = np.linspace(xmin, xmax, 1000)
    probs = st.norm.pdf(x, mu_hat, sigma_hat / np.sqrt(N))
    label = f"variant {variant}: {round(100. * mu_hat, 2)}%"
    ax.plot(x, probs, label=label)


def plot_timeline_experiments(
    df,
    labels,
    step,
    ax=None,
    plot_significance=True,
    plotting_alpha=1.0,
    sample_size=None,
    **experiment_args,
):

    # plot p-value over time
    if not ax:
        f, ax = plt.subplots(figsize=(9, 7))

    df = df.copy()
    df = df[df.variant.isin(labels)]

    results = {"samples_so_far": [], "statistic": [], "pvalue": [], "passed": []}
    samples = df.shape[0]
    experiment_args["verbose"] = experiment_args.get("verbose", False)

    # run experiments over time
    for i in range(0, samples, step - 1):
        _df = df.iloc[:i]
        if _df.groupby("variant").count().shape[0] == 2:
            experiment = FrequentistExperiment(_df, labels=labels, **experiment_args)
            result = experiment.run()
            results["samples_so_far"].append(i + 1)
            for k, v in result.items():
                results[k].append(v)
    df_results = pd.DataFrame(data=results)

    # significance
    alpha = experiment.significance

    # plot sample_size if given
    if sample_size:
        # we plot 2 * sample_size because we're looking at the whole experiment
        ax.axvline(
            2 * sample_size,
            label=f"sample size: {sample_size}",
            linestyle="--",
            color="black",
            alpha=0.7,
        )

    sns.lineplot(
        data=df_results, x="samples_so_far", y="pvalue", ax=ax, alpha=plotting_alpha
    )
    if plot_significance:
        ax.axhline(
            alpha,
            label=f"significance: {alpha}",
            linestyle="--",
            color="red",
            alpha=0.7,
        )
    ax.set_xlabel("Sample size so far", fontsize=15)
    ax.set_ylabel("p-value", fontsize=15)

    plt.legend()

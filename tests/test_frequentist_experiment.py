from numpy import absolute
from abtests.frequentist_experiment import *


def test_generate_binomial_data():
    df = generate_binomial_data()
    assert df.shape == (200, 2)
    assert df.columns.tolist() == ["variant", "target"]


def test_estimate_sample_size_1():
    n = estimate_sample_size(min_diff=0.05, mu_baseline=0.20, test_type="two-sided")
    assert n == 1030


def test_estimate_sample_size_2():
    n = estimate_sample_size(
        min_diff=0.05, mu_baseline=0.20, effect_type="relative", test_type="two-sided"
    )
    assert n == 25255

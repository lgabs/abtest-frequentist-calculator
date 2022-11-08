from abtests.frequentist_experiment import *
from abtests.frequentist_experiment import estimate_sample_size
from utils import help_strings

import streamlit as st

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

st.set_page_config(page_title="Test_Planning", page_icon="📊")

st.markdown("# Sample Size Calculator")
st.markdown(
    "Before running your test, make sure you'll have enough data. Fill in the following parameters (use the '?' tooltip symbol for details):"
)
with st.form(key="my_form"):
    min_diff = st.number_input(
        label="minimum detectable effect you want to see (relative or absolute is the next parameter)",
        value=0.05,
        help="If you are expecting to increase your metric by 5%, this is the minimum detectable effect you have to choose.",
    )
    effect_type = st.selectbox(
        label="Effect Type of the previous parameter",
        options=["absolute", "relative"],
        help="Example: if you expect to increase your metric from 10% to 5%, this is a 50% relative increase and a 5% absolute increase.",
    )
    mu_baseline = st.number_input(
        label="Metric commom value for baseline variant",
        value=0.20,
        help="Ex: if you're studying conversions and you current baseline variant converts 20%, chose 20% here. If you're studying ARPU and your typical value is R$ 0.12, chose this value.",
    )
    test_type = st.selectbox(
        label="Test type (right-sided or two-sided)",
        options=["right-sided", "two-sided"],
        help=help_strings["test_type"],
    )
    test_type = test_type if test_type == "two-sided" else "one-sided"
    objective_metric_type = st.selectbox(
        label="Objective Metric Type",
        options=["binary", "continuous"],
        help=help_strings["objective_metric_type"],
    )

    logging.info(f"objective_metric_type = {objective_metric_type}")

    estimated_impressions_daily = int(
        st.number_input(
            label="(optional) number of impressions daily per variant estimated a priori",
        )
    )

    submit_button = st.form_submit_button(label="Calculate Sample Size")

    if submit_button:
        estimate_sample_size(
            min_diff=min_diff,
            effect_type=effect_type,
            mu_baseline=mu_baseline,
            test_type=test_type,
            estimated_impressions_daily=estimated_impressions_daily,
            streamlit_print=True,
            objective_metric_type=objective_metric_type,
        )


st.markdown("# Plot sample sizes for different minimum expected differences")
st.markdown(
    "To choose some miminum detectable effect consistent with your traffic, you can plot a line that gives you the sample size for each minimum detactable effect. Fill in the following parameters:"
)

with st.form(key="my_form2"):
    effect_type = st.selectbox(
        label="Effect Type of the previous parameter", options=["absolute", "relative"]
    )
    mu_baseline = st.number_input(
        label="Metric commom value for baseline variant", value=0.20
    )
    test_type = st.selectbox(
        label="Test type (right-sided or two-sided)",
        options=["two-sided", "right-sided"],
        help=help_strings["test_type"],
    )
    test_type = test_type if test_type == "two-sided" else "one-sided"

    objective_metric_type = st.multiselect(
        label="Objective Metric Type",
        options=["binary", "continuous"],
        default="binary",
        help="Choose some type for your metric. If it is a yes/no metric, with only two outcomes, choose 'binary'. If it's continous outcome, like revenue, cost, time in seconds, choose 'continuous'.",
    )[0]

    min_diff_min = st.number_input(
        label="Lower bound for the minimum expected difference", value=0.01
    )
    min_diff_max = st.number_input(
        label="Upper bound for the minimum expected difference", value=0.05
    )
    min_diff_range = np.linspace(min_diff_min, min_diff_max, 100)

    submit_button2 = st.form_submit_button(label="Plot Sample Sizes")

if submit_button2:
    plot_sample_sizes(
        min_diff_range=min_diff_range,
        effect_type=effect_type,
        mu_baseline=mu_baseline,
        streamlit_plot=True,
        test_type=test_type,
        objective_metric_type=objective_metric_type,
    )

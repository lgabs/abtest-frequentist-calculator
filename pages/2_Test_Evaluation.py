from abtests.frequentist_experiment import *
from utils import help_strings, make_per_user_dataset

import streamlit as st

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

st.set_page_config(page_title="Test_Evaluation", page_icon="📊")

st.markdown(
    """
# 📊 A/B Testing Evaluation

Evaluate your A/B Test filling the parameters below. See help comments in the '?' tooltip in each parameter.

Basic parameters are:
- **total impressions in control/treatment**: total of participants in each variant.
- **total conversions in control/treatment**: total of conversions in each variant.
- **total conversion value in control/treatment**: the sum of conversion values in each variant (e.g.: revenue, cost).
- **Objective Metric Type:** You can only analyze one Objective Metric at a time. Choose 'binary' for binary outcomes like conversions (yes/no question). Choose 'continuous' for outcomes like revenue, cost etc. 
"""
)

# Sample Size information
st.write(
    """
#### Sample Size information
Optional, but **strongly recommended** to fill.
"""
)

use_sample_size = st.checkbox(label="use sample size", value=False)
if use_sample_size:
    sample_size_estimated = st.number_input(label="Estimated Sample Size", step=1)
    mu_baseline = st.number_input(
        label="Baseline value for you metric so far",
        help="e.g.: if your conversion is currently 10% in you context, fill with 0.10.",
    )
else:
    sample_size_estimated = None
    mu_baseline = None

with st.form(key="my_form"):
    st.write("### Basic information")
    # Control
    control_impressions = st.number_input(
        label="Impressions in Control",
        value=1000,
    )
    control_conversions = st.number_input(
        label="Conversions in Control",
        value=100,
    )
    control_total_value = st.number_input(
        label="Total Conversion Value in Control",
        value=200,
        help="Total value that came from control, e.g.: put 200 if control group generated R$ 200.",
    )

    # Treatment
    treatment_impressions = st.number_input(
        label="Impressions in Treatment",
        value=1000,
    )
    treatment_conversions = st.number_input(
        label="Conversions in Treatment",
        value=120,
    )
    treatment_total_value = st.number_input(
        label="Total Conversion Value in Treatment",
        value=250,
        help="Total value that came from control, e.g.: put 250 if treatment group generated R$ 250.",
    )

    # Advanced information
    st.write("### Advanced information")
    test_type = st.selectbox(
        label="Test type (right-sided or two-sided)",
        options=["right-sided", "two-sided"],
        help=help_strings["test_type"],
    )
    objective_metric_type = st.selectbox(
        label="Objective Metric Type",
        options=["binary", "continuous"],
        help=help_strings["objective_metric_type"],
    )
    confidence = st.number_input(
        label="Confidence (typically 95%)", value=95, min_value=0, max_value=100
    )
    significance = round(1 - confidence / 100, 2)
    power = (
        st.number_input(label="Power of test", value=80, min_value=0, max_value=100)
        / 100
    )

    # Submit
    submit_button = st.form_submit_button(label="Run Experiment")

    if submit_button:
        # prepare options to run experiement analysis
        test_dist = (
            "proportions-ztest" if objective_metric_type == "binary" else "ztest"
        )
        labels = ["control", "treatment"]
        alternative = (
            "larger" if test_type in ("right-sided", "left-sided") else "two-sided"
        )

        # build dataset using summary info
        df = make_per_user_dataset(
            control_impressions=control_impressions,
            treatment_impressions=treatment_impressions,
            control_conversions=control_conversions,
            treatment_conversions=treatment_conversions,
            control_total_value=control_total_value,
            treatment_total_value=treatment_total_value,
            objective_metric_type=objective_metric_type,
        )

        # st.dataframe(df.sample(frac=1.0))

        # run experiment
        # Initialize Experiment
        with st.spinner(f"Analyzing Experiment..."):
            experiment = FrequentistExperiment(
                df=df,
                test_dist=test_dist,
                alternative=alternative,
                significance=significance,
                power=power,
                labels=["control", "treatment"],
            )
            if mu_baseline and sample_size_estimated:
                logging.info("Filling sample size parameters...")
                experiment.update_minimum_sample_size(sample_size_estimated)
                experiment.update_mu_baseline(mu_baseline)
            experiment.run(should_print_streamlit_report=True)

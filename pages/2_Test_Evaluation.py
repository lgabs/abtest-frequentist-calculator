from abtests.frequentist_experiment import *

import streamlit as st

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

st.set_page_config(page_title="Test_Evaluation", page_icon="📊")
st.sidebar.markdown("# Test Evaluation")

st.markdown(
    """
# 📊 A/B Testing Evaluation

Evaluate your A/B Test filling these parameters:

- **total impressions in control/treatment**: total of participants in each variant.
- **total observed metric in control/treatment**: total of observed metric in each variant. Common example is observed conversions, but you can analyze revenue per conversion or _Average Revenue Per (ARPU)_ User too.
- **objective metric type: type of the objective metric. It must be 'binomial' for conversion metric and 'non-binomial' for monetary values like revenue or ARPU.**
"""
)

with st.form(key="my_form"):
    # Control
    control_impressions = st.number_input(
        label="Impressions in Control",
        value=1000,
    )
    control_observed_metric = st.number_input(
        label="Observed Metric in Control",
        value=100,
    )

    # Treatment
    test_impressions = st.number_input(
        label="Impressions in Treatment",
        value=1000,
    )
    test_observed_metric = st.number_input(
        label="Observed Metric in Treatment",
        value=120,
    )

    # More params
    test_type = st.selectbox(
        label="Test type (one-sided or two-sided)",
        options=["two-sided", "one-sided"],
    )

    experiment_name = st.text_input(
        label="Experiment Name (Optional)", value="My Experiment"
    )

    submit_button = st.form_submit_button(label="Run Experiment")

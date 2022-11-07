from abtests.frequentist_experiment import *

from abtests.frequentist_experiment import estimate_sample_size
import streamlit as st

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

st.set_page_config(page_title="Test_Planning", page_icon="📊")

st.sidebar.markdown("# Test Planning")

st.markdown("# Sample Size Calculator")
st.markdown(
    "Before running your test, make sure you'll have enough data. Fill in the following parameters (use the 'help' tooltip for details):"
)
with st.form(key="my_form"):
    min_diff = st.number_input(
        label="minimum detectable effect you want to see (relative or absolute is the next parameter)",
        value=0.05,
    )
    effect_type = st.selectbox(
        label="Effect Type of the previous parameter", options=["absolute", "relative"]
    )
    mu_baseline = st.number_input(
        label="Conversion rate for baseline variant (0 to 1.0)", value=0.20
    )
    test_type = st.selectbox(
        label="Test type (one-sided or two-sided)",
        options=["two-sided", "one-sided"],
    )
    objective_metric_type = st.multiselect(
        label="Objective Metric Type",
        options=["binary", "continuous"],
        default='binary',
        help="Choose some type for your metric. If it is a yes/no metric, with only two outcomes, choose 'binary'. If it's continous outcome, like revenue, cost, time in seconds, choose 'continuous'.",
    )[0]
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
        label="Conversion rate for baseline variant (0 to 1.0)", value=0.20
    )
    test_type = st.selectbox(
        label="Test type (one-sided or two-sided)",
        options=["two-sided", "one-sided"],
    )
    objective_metric_type = st.multiselect(
        label="Objective Metric Type",
        options=["binary", "continuous"],
        default='binary',
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

st.write(
    """
## Other references:
- [calculator evan miller](https://www.evanmiller.org/ab-testing/sample-size.html). Big reference.
- [ab test guide](https://abtestguide.com/abtestsize/), calculations differ by little (sometimes calculators use rules of thumb).
- [discussion in YC's Forum (advanced)](https://news.ycombinator.com/item?id=13437431)
- [Sample Sizes Required - Suny Polytechnich](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm) 
- [How to calculate ab testing sample size - Stack Overflow](https://stackoverflow.com/questions/28046453/how-to-calculate-ab-testing-sample-size). Here I found a book about rules of thumb that follows
- [Statistical Rules of Thumb](http://www.vanbelle.org/struts.htm) homepage. [Link to download](http://library.lol/main/3306598CAA57137F059CFC4875A4230F).
- [another calculator - CXL](https://cxl.com/ab-test-calculator/)
"""
)

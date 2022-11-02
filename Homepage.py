import streamlit as st

import logging

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="A/B Test Frequentist App", page_icon="📊")

st.sidebar.markdown("# Main Page")

st.markdown(
    """
# 📊 A/B Test Frequentist App

This is a tool to automate your decisions in an A/B Test Experiment with two alternatives, where 
you want to know which one is better¹. It is focused on conversion and revenue evaluation, typical in e-commerce. It uses Frequentist Statistics.

## How to use

An A/B Test has two phases: test planning and test evaluation. In the side bar you can find the section you need.

- Test Planning: before running your A/B Test, make sure to plan the sample size needed to measure the effect you want.
- Test Evaluation: after running your test, use this section to evaluate the results.


¹ To analyze more than one alternative, you can approximately still compare 
them two by two, fixing you alpha level to alpha/n, where n is the number of variants (Bonferroni Correction).

² See references and more at my [github](https://github.com/lgabs/abtest-calculator).
"""
)

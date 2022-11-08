from typing import Union
import pandas as pd

help_strings = {
    "test_type": "This is tricky. Choose 'right-side' when you want to study if B is better than A (most frequent use case). If you want to study if B is different than A (higher or lower), choose 'two-sided'.",
    "objective_metric_type": "Choose some type for your metric. If it is a yes/no metric like conversions, choose 'binary'. If it's continous outcome, like revenue, cost, time in seconds, choose 'continuous'.",
}


def make_variant_per_user_dataset_binomial(n: int, c: int, label: str):
    """
    Makes a per-user dataset for binomial metrics (like conversion).
    """
    return pd.DataFrame(
        data={
            "variant": [label] * n,
            "target": [0] * (n - c) + [1] * c,
        }
    )


def make_variant_per_user_dataset_continuous(
    n: int, c: int, s: Union[int, float], label: str
):
    """
    Makes a per-user dataset where conversion values are total value is equally distributed between converted impressions.
    """
    return pd.DataFrame(
        data={
            "variant": [label] * n,
            "target": [0] * (n - c) + [s / c] * c,
        }
    )


def make_per_user_dataset(
    control_impressions: int,
    treatment_impressions: int,
    control_conversions: int,
    treatment_conversions: int,
    control_total_value: Union[int, float],
    treatment_total_value: Union[int, float],
    objective_metric_type: str,
) -> pd.DataFrame:
    """
    Build a dataset equivalent to the summary data informed.
    """
    if objective_metric_type == "binary":
        df_control = make_variant_per_user_dataset_binomial(
            control_impressions, control_conversions, "control"
        )
        df_treatment = make_variant_per_user_dataset_binomial(
            treatment_impressions, treatment_conversions, "treatment"
        )
        df = pd.concat([df_control, df_treatment], axis=0)
    elif objective_metric_type == "continuous":
        # in this case, for each conversion we assign an average of total values
        df_control = make_variant_per_user_dataset_continuous(
            control_impressions, control_conversions, control_total_value, "control"
        )
        df_treatment = make_variant_per_user_dataset_continuous(
            treatment_impressions,
            treatment_conversions,
            treatment_total_value,
            "treatment",
        )
        df = pd.concat([df_control, df_treatment], axis=0)

    else:
        raise ValueError(
            "It's only possible to make per-user dataset for objective_metric_type equals to 'binary' or 'continuous'."
        )
    # check values
    assert df_control.shape[0] == control_impressions
    assert df_treatment.shape[0] == treatment_impressions
    if objective_metric_type == "continuous":
        assert (
            df_control[df_control.target > 0].target.sum() == control_total_value
        ), f"{df_control[df_control.target > 0].target.sum()} != {control_total_value}"
        assert (
            df_treatment[df_treatment.target > 0].target.sum() == treatment_total_value
        )
    else:
        assert (
            df_control[df_control.target > 0].target.sum() == control_conversions
        ), f"{df_control[df_control.target > 0].target.sum()} != {control_conversions}"
        assert (
            df_treatment[df_treatment.target > 0].target.sum() == treatment_conversions
        ), f"{df_treatment[df_treatment.target > 0].target.sum()} != {treatment_conversions}"

    return df

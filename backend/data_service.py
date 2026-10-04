import pandas as pd

REQUIRED_COLUMNS = [
    "v_0h", "i_0h", "temp_0h",
    "v_24h", "i_24h", "temp_24h",
    "actual_failure_168h"
]


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df = df.copy()

    # Create early drift features automatically.
    df["delta_v_0_24"] = df["v_24h"] - df["v_0h"]
    df["delta_i_0_24"] = df["i_24h"] - df["i_0h"]
    df["delta_temp_0_24"] = df["temp_24h"] - df["temp_0h"]

    return df

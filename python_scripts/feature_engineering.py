import pandas as pd


def engineer_features(df, target_col="carbon_intensity__gCO2_kWh", add_year_lag=False):
    """
    Create lag and calendar features for modelling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a datetime column and target column.

    target_col : str, default="carbon_intensity"
        Name of the target column to create lag features from.

    add_year_lag : bool, default=False
        Whether to add a 1-year lag feature.
        Only use this if the dataframe contains at least 2 years of half-hourly data.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered lag and calendar features.
    """

    df = df.copy()

    # Ensure datetime is in datetime format
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Drop rows with invalid datetime values
    df = df.dropna(subset=["datetime"])

    # Sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)

    # Lag features
    # 48 half-hour periods = 24 hours
    df["lag_48"] = df[target_col].shift(48)

    # 336 half-hour periods = 7 days
    df["lag_336"] = df[target_col].shift(336)

    # Optional yearly lag
    # 17520 half-hour periods = 365 days
    if add_year_lag:
        df["lag_336"] = df[target_col].shift(17520)

    # Calendar features
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


def validate_features(df):
    """
    Print validation checks for the engineered dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe.

    Returns
    -------
    None
    """

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nDuplicate datetimes:")
    if "datetime" in df.columns:
        print(df["datetime"].duplicated().sum())
    else:
        print("datetime column not found")

    print("\nDatetime spacing:")
    if "datetime" in df.columns:
        print(df["datetime"].diff().value_counts().head())
    else:
        print("datetime column not found")


def drop_lag_nulls(df):
    """
    Drop rows with null values created by lag features.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with null rows removed and index reset.
    """

    return df.dropna().reset_index(drop=True)

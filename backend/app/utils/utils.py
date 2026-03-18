import pandas as pd

def get_day_from_forecast(cleaned_df: pd.DataFrame, target_date):
    """
    Slices a 24-hour block from the pre-cleaned 14-day DataFrame.
    """
# 1. Convert input to a standard date object
    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date).date()
    elif hasattr(target_date, 'date'): # Handles datetime objects
        target_dt = target_date.date()
    else:
        target_dt = target_date # Already a date object

    # Filter the DataFrame
    # Note: This assumes your weather_preproc sets 'time' as the Index
    day_data = cleaned_df[cleaned_df.index.date == target_dt]

# 3. Validation
    if day_data.empty:
        raise ValueError(f"Date {target_dt} is out of the 14-day forecast range.")

    if len(day_data) != 24:
        # Useful if the API returned a partial day at the start/end of the window
        raise ValueError(f"Found {len(day_data)} hours for {target_dt}, but model requires 24.")

    return day_data

#CHECK IF THIS FORMAT WORKS

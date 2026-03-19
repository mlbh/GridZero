import requests
import pandas as pd
from datetime import timedelta, datetime


def fetch_carbon_history():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fmt = "%Y-%m-%dT%H:%MZ"

    # yesterday
    y_start = today - timedelta(days=1)
    y_end = y_start + timedelta(hours=23, minutes=30)

    # year ago
    ya_start = today - timedelta(days=364)
    ya_end = ya_start + timedelta(days=13, hours=23, minutes=30)

    def get_data(s, e):

        url = f"https://api.carbonintensity.org.uk/intensity/{s.strftime(fmt)}/{e.strftime(fmt)}"

        r = requests.get(url)
        r.raise_for_status()



        return [item['intensity']['actual'] for item in r.json()['data']]

    return {
    "yesterday": get_data(y_start, y_end),
    "year_ago": get_data(ya_start, ya_end)
    }

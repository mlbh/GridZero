import requests
from datetime import datetime, timedelta

# The address where your uvicorn is running
BASE_URL = "http://127.0.0.1:8000"

def test_root():
    response = requests.get(f"{BASE_URL}/")
    print(f"Root Test: {response.status_code} - {response.json()}")

def test_prediction(days_ahead=2):
    # Create a date within the 14-day window
    target_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    payload = {
        "target_date": target_date
    }

    print(f"\nSending prediction request for: {target_date}...")
    response = requests.post(f"{BASE_URL}/predict", json=payload)

    if response.status_code == 200:
        data = response.json()
        print("✅ Success!")
        print(f"Date returned: {data['date']}")
        print(f"Gen Predictions (first 3): {data['generation_prediction'][:3]}")
        print(f"Carbon Intensity (first 3): {data['carbon_intensity'][:3]}")

        # Verify we got 24 hours of data
        if len(data['generation_prediction']) == 24:
            print("✅ Data integrity check: Received 24 hourly predictions.")
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"Error Detail: {response.text}")

def test_out_of_range():
    # Test a date 30 days in the future (should trigger our ValueError/400)
    target_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    print(f"\nTesting out-of-range date: {target_date}...")

    response = requests.post(f"{BASE_URL}/predict", json={"target_date": target_date})
    if response.status_code == 400:
        print("✅ Correctly caught out-of-range error.")
    else:
        print(f"❌ Unexpected behavior: Got {response.status_code}")

if __name__ == "__main__":
    try:
        test_root()
        test_prediction(days_ahead=1) # Test tomorrow
        test_prediction(days_ahead=5) # Test later this week
        test_out_of_range()           # Test the error handling
    except requests.exceptions.ConnectionError:
        print("❌ Error: Is your FastAPI server running? (uvicorn main:app --reload)")

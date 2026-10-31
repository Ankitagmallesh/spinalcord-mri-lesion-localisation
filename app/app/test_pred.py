import requests
from pathlib import Path

url = "http://localhost:8000/api/predict"
file_path = Path(__file__).parent / "frontend" / "src" / "assets" / "test_tumor.png"
if not file_path.exists():
    file_path = Path(__file__).parent / "ml" / "data" / "Tumor" / "images" / "Tumor_000.png"

try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
        
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print(response.text)
    else:
        print("Success!")
except Exception as e:
    print(f"Error: {e}")

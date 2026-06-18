code to run the backend : uvicorn backend:app --reload --host 0.0.0.0 --port 8000  
code to run the front end: streamlit run frontend.py

Sample code if would like to upload conversation via a csv file:

import requests
import pandas as pd

df = pd.read_csv("conv.csv")

BACKEND_URL = "http://localhost:8000"

for conversation in df['conversation']:
    response = requests.post(
            f"{BACKEND_URL}/evaluate",
            json={
                "conversation":
                conversation
            }
        )

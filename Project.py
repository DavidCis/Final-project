import requests
import pandas as pd
import numpy as np
import json

API_KEY = 'tYTvFTiccRfj'
PROJECT_TOKEN = 'tRhVHZCkQ_BC'
RUN_TOKEN  = 'tQk29TnXY0u6'

r = requests.get(f'https://www.parsehub.com/api/v2/projects/{PROJECT_TOKEN}/last_ready_run/data', params={'api_key':API_KEY})
#data = json.loads(r.text)

h1=pd.read_csv('H1N1.csv')
print(h1.head())
import requests
import json

API_KEY = 'tYTvFTiccRfj'
PROJECT_TOKEN = 'tTXGYJn0fKOF'
RUN_TOKEN = 'tO1VzbSeYdh6'

class Data:
    def __init__(self,api_key,project_token):
        self.api_key = api_key
        self.project_token = project_token 
        self.params = {
            'api_key':self.api_key
        }
        self.get_data()


    def get_data(self): #get the data from the API using token
        res = requests.get(f'https://www.parsehub.com/api/v2/projects/{PROJECT_TOKEN}/last_ready_run/data', params={'api_key':API_KEY})
        self.data = json.loads(res.text)


    def get_today_price(self): #gets real time oil price
        return self.data['price'] 

    
    def get_date_price(self,date): #gets oil price and var from the lasth month
        data=self.data['last_month']

        for c in data:
            if c['date'] == date:
                return c

        return '0'        


data = Data(API_KEY,PROJECT_TOKEN)
print(data.get_today_price())
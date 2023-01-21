import requests

#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

url = 'https://f97swtv3n3.execute-api.us-east-1.amazonaws.com/test/predict'

data = {'url': 'https://bit.ly/3IYiPAX'}

result = requests.post(url, json=data).json()
print(result)



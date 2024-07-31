import ollama
import requests
import json

def get_stock_data(ticker):
    api_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=1min&apikey=demo"
    response = requests.get(api_url)
    data = response.json()
    
    if "Time Series (1min)" in data:
        latest_time = max(data["Time Series (1min)"].keys())
        latest_data = data["Time Series (1min)"][latest_time]
        return {
            "time": latest_time,
            "open": latest_data["1. open"],
            "high": latest_data["2. high"],
            "low": latest_data["3. low"],
            "close": latest_data["4. close"],
            "volume": latest_data["5. volume"]
        }
    else:
        return {"error": "Unable to fetch stock data"}

response = ollama.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': 'What is the latest stock data for IBM?'}],
    tools=[{
        'type': 'function',
        'function': {
            'name': 'get_stock_data',
            'description': 'Get the latest intraday stock data for a given ticker symbol',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'The ticker symbol of the stock',
                    },
                },
                'required': ['ticker'],
            },
        },
    }],
)

print("Tool Calls:")
print(json.dumps(response['message']['tool_calls'], indent=2))

# Extract and use arguments
tool_call = response['message']['tool_calls'][0]
if tool_call['function']['name'] == 'get_stock_data':
    ticker = tool_call['function']['arguments']['ticker']
    result = get_stock_data(ticker)
    print("\nStock Data:")
    print(json.dumps(result, indent=2))

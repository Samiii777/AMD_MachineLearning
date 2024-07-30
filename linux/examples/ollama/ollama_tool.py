import ollama
import requests
from rich import print
from datetime import datetime

def get_current_weather(city):
    base_url = f"http://wttr.in/{city}?format=j1"
    response = requests.get(base_url)
    data = response.json()
    return f"The current temperature in {city} is: {data['current_condition'][0]['temp_C']}°C"

def get_current_time():
    current_time = datetime.now()
    return f"The current time is: {current_time.strftime('%I:%M %p')}"

response = ollama.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content':
        'What is the weather in Toronto and what time is it?'}],

    tools=[
        {
            'type': 'function',
            'function': {
                'name': 'get_current_weather',
                'description': 'Get the current weather for a city',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'city': {
                            'type': 'string',
                            'description': 'The name of the city',
                        },
                    },
                    'required': ['city'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_current_time',
                'description': 'Get the current time',
                'parameters': {
                    'type': 'object',
                    'properties': {},
                },
            },
        },
    ],
)

tools_calls = response['message']['tool_calls']

results = []
for tool_call in tools_calls:
    tool_name = tool_call['function']['name']
    arguments = tool_call['function']['arguments']

    if tool_name == 'get_current_weather':
        city = arguments['city']
        results.append(get_current_weather(city))
    elif tool_name == 'get_current_time':
        results.append(get_current_time())

for result in results:
    print(result)

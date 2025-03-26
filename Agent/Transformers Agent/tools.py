import requests
from tavily import TavilyClient
from transformers.agents.tools import Tool
from configs import TAVILY_API, OPENWEATHER_API

tavily_client = TavilyClient(api_key=TAVILY_API)


class WeatherTool(Tool):
    name = "weather"
    description="""This tool provides local weather information.
    'weather' indicates the weather conditions. 'main' contains the core physical information of weather, such as temperature, pressure, and humidity. 'visibility' contains visibility, 'wind' contains wind speed and direction information, and 'clouds' is the amount of clouds, with 0 being clear and 100 being cloudy.
    The basic unit of temperature is Kelvin."""
    inputs = {"city_name":{"type":"string", "description":"This is the name of the region. It must be entered in English."}}
    output_type = "any"

    def forward(self, *args, **kwargs):
        city_name = kwargs.get("city_name")
        if not city_name:
            if args:
                city_name = args[0]
            else:
                raise ValueError("city_name input is required")
        url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={OPENWEATHER_API}'
        response = requests.get(url)
        weather_data = response.json()
        return weather_data

    

class WebSearchTool(Tool):
    name = "web-search"
    description="""This tool is a function that provides content searched on the web.
    When the user's request requires the latest information or is a complex request, this tool is used to search for information on the web page and provide the content."""
    inputs = {"query":{"type":"string", "description":"The user's question should be in the form of a keyword or a brief summary sentence that can be easily searched on the web."}}
    output_type = "any"

    def forward(self, *args, **kwargs):
        query = kwargs.get("query")
        if not query:
            if args:
                query = args[0]
            else:
                raise ValueError("query input is required")

        try:
            results = tavily_client.search(query)
        except Exception as e:
            return f"An error occurred during the search: {str(e)}"
        return results
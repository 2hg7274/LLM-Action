import os
import psutil
import json
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
    


class TopProcessesByMemoryTool(Tool):
    name = "top-processes-by-memory"
    description = """This tool retrieves the top 10 running processes sorted in descending order by their memory usage percentage.
    It uses psutil to gather process information and returns the results in JSON format."""
    inputs = {}  # No inputs are required for this tool.
    output_type = "any"

    def forward(self, *args, **kwargs):
        processes = []
        # Iterate over all running processes with selected attributes.
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # Sort processes by memory_percent in descending order.
        processes = sorted(processes, key=lambda proc: proc.get('memory_percent', 0), reverse=True)
        top_processes = processes[:10]

        # Return the top 30 processes in a pretty-printed JSON format.
        return json.dumps(top_processes, indent=2)
    


class ProcessKillerTool(Tool):
    name = "process-killer"
    description = """This tool terminates a Linux program given its process ID (PID).
    It uses psutil to locate the process and sends a termination signal.
    If the process does not exit gracefully, it forcefully kills the process."""
    inputs = {"pid": {"type": "integer", "description": "The process ID (PID) of the Linux program to terminate."}}
    output_type = "any"

    def forward(self, *args, **kwargs):
        # Retrieve the PID from the input arguments
        pid = kwargs.get("pid")
        if pid is None:
            if args:
                pid = args[0]
            else:
                raise ValueError("The 'pid' input is required.")

        # Prevent terminating the current running process
        current_pid = os.getpid()
        if pid == current_pid:
            return f"Cannot terminate the currently running process (PID {current_pid})."

        try:
            # Attempt to get the process with the specified PID
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return f"No process found with PID {pid}."

        try:
            # First, try to terminate the process gracefully
            process.terminate()
            process.wait(timeout=3)
        except psutil.TimeoutExpired:
            # If the process does not exit gracefully, force kill it
            process.kill()
            process.wait(timeout=3)
        except Exception as e:
            return f"An error occurred while terminating the process: {str(e)}"

        return f"Process with PID {pid} has been terminated successfully."
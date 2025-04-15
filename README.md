# LLM Action Project

## Overview
The LLM Action project is designed to facilitate interactions with language models and external tools for various tasks, including weather information retrieval, system monitoring, and multi-agent workflows.

## Directory Structure
```
/home/user/2HG/LLM-Action
├── Function Calling
│   ├── OpenAI function calling
│   │   ├── functions.py
│   │   └── openai_functioncalling.py
│   └── Transformers function calling
│       ├── functions.py
│       ├── custom_system_prompt.py
│       └── transformers_functioncalling.py
└── Agent
    ├── llama_index
    │   ├── agent.py
    │   ├── agent_workflow_multi.ipynb
    │   ├── tools.py
    │   └── workflow_agent.py
    └── Transformers Agent
        ├── transformers_agent.py
        └── custom_prompt.py
```

## Function Calling

### OpenAI Function Calling

#### `functions.py`
This file provides utility functions related to weather information and location nicknames.
- **get_current_weather(location: str, unit: str)**: Fetches the current weather for a specified location.
- **get_location_nickname(location: str)**: Generates a nickname for a given location.

#### `openai_functioncalling.py`
This file serves as a bridge between user input and the OpenAI API.
- **make_function_list()**: Creates a list of available functions and their schemas.
- **main(user_input, functions, avail_functions, chat_list)**: Manages chat interactions and function calls.

### Transformers Function Calling

#### `transformers_functioncalling.py`
This file serves as an interface between user input and a transformer model.
- **make_function_list()**: Creates a list of available functions and their schemas.
- **load_model(path)**: Loads a tokenizer and model for generating responses.
- **gen(model, tokenizer, chat_list)**: Generates a response from the model based on chat history.

#### `custom_system_prompt.py`
Defines a system prompt that instructs the chatbot on how to handle tool calls.

## Agent

### `agent.py`
Defines an agent that utilizes a language model and tools to interact with users and perform tasks related to server log data analysis.
- **load_tools()**: Loads tools for managing system processes.
- **make_agent(path)**: Initializes a HuggingFace language model and creates a `ReActAgent`.

### `agent_workflow_multi.ipynb`
Explores how to use the `AgentWorkflow` class to create multi-agent systems.
- **Setup**: Uses `OpenAI` as the LLM for all agents.
- **System Design**: Consists of three agents: `ResearchAgent`, `WriteAgent`, and `ReviewAgent`.

### `tools.py`
Contains various utility functions for system monitoring and management.
- **get_top_processes_by_memory()**: Retrieves the top 10 running processes by memory usage.
- **kill_process(pid: int)**: Terminates a Linux program given its process ID.

### `transformers_agent.py`
Defines an agent that utilizes a transformer model to interact with users.
- **make_pipeline(path)**: Loads a tokenizer and model for generating responses.
- **make_tools()**: Initializes tools for web searching and managing processes.

## Conclusion
The LLM Action project provides a comprehensive framework for interacting with language models and external tools, enabling efficient task execution and multi-agent workflows.
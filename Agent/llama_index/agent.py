import os
from llama_index.core.agent import ReActAgent
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.tools import FunctionTool
from tools import get_top_processes_by_memory, kill_process
os.environ["CUDA_VISIBLE_DEVICES"] = "1"




def load_tools():
    top_processes_tool = FunctionTool.from_defaults(fn=get_top_processes_by_memory)
    kill_process_tool = FunctionTool.from_defaults(fn=kill_process)
    return [top_processes_tool, kill_process_tool]


def make_agent(path):
    llm = HuggingFaceLLM(
        model_name=path,
        tokenizer_name=path,
        max_new_tokens=4096,
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        device_map="auto",
    )

    tools = load_tools()
    context = (
        "Here is a description of the server log data:"
        "- timestamp: The time at which the data was collected or predicted (ISO format string)"
        "- cpu_usage_percent: The CPU usage as a percentage"
        "- memory_usage_percent: The memory usage as a percentage"
        "- disk_usage_percent: The disk usage as a percentage"
        "- confidence: The confidence of the model in the prediction result (a value between 0 and 1, with values ​​closer to 1 indicating higher confidence)"
        "- error_occur: Whether the data at that point in time predicted an error (True or False)"
        "Given the log data, use tools to troubleshoot the error."
    )

    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        context=context,
        verbose=True
    )
    return agent




if __name__=="__main__":
    from configs import MODEL_PATH
    agent = make_agent(MODEL_PATH)
    while True:
        user_input = input("USER: ")
        if user_input.strip().upper() == "END":
            print("CHAT HISTORY")
            print(agent.chat_history)
            break
        response = agent.chat(user_input)
        print("AGENT:", response)

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
        "You are an intelligent agent capable of engaging in general conversation and system management tasks. "
        "If a user asks for system process information using keywords such as 'process', 'memory usage', or 'top processes', "
        "you must use the get_top_processes_by_memory tool to retrieve and return the top 10 processes in JSON format, "
        "sorted by memory usage. "
        "If a user requests to terminate a process by providing its process ID (PID), "
        "you must use the kill_process tool to safely terminate the specified process (ensuring you do not terminate the current process)."
    )

    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        context=context,
        verbose=True
    )
    return agent

# 6. 사용자 입력을 받아 "END"가 입력될 때까지 반복 실행


if __name__=="__main__":
    from configs import MODEL_PATH
    agent = make_agent(MODEL_PATH)
    while True:
        user_input = input("User: ")
        if user_input.strip().upper() == "END":
            print("CHAT HISTORY")
            print(agent.chat_history)
            break
        response = agent.chat(user_input)
        print("Agent:", response)

import os
import time
import warnings
from transformers.agents.llm_engine import TransformersEngine
from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    ReactJsonAgent
)
from tools import WebSearchTool, TopProcessesByMemoryTool, ProcessKillerTool

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings(action='ignore')






def make_pipeline(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype="auto"
    )
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        device_map="auto"
    )

    return pipe

def make_tools():
    # web_search_tool = WebSearchTool()
    execute_tool = TopProcessesByMemoryTool()
    process_killer_tool = ProcessKillerTool()
    return [execute_tool, process_killer_tool]

def make_agent(pipe, system_prompt):
    llm_engine = TransformersEngine(pipeline=pipe)
    llm_engine.tokenizer = llm_engine.pipeline.tokenizer

    tools = make_tools()

    system_prompt+="\n\n"
    agent = ReactJsonAgent(
        tools=tools,
        llm_engine=llm_engine,
        max_iterations=10,
        system_prompt=system_prompt
    )
    return agent





def main(user_input, chat_list, agent, chat_flag):
    if len(chat_list) == 0:
        chat_list.append({"role":"system", "content":agent.system_prompt})

    chat_list.append({"role":"user", "content":user_input})
    tokenized_user_input = agent.llm_engine.tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=True)

    if chat_flag:
        output = agent.run(tokenized_user_input)
    else:
        output = agent.run(tokenized_user_input, reset=False)
    chat_list.append({"role":"assistant", "content":output})
    return chat_list 




if __name__=="__main__":
    from configs import LLM_MODEL_PATH
    from custom_prompt import DEFAULT_REACT_JSON_SYSTEM_PROMPT

    pipe = make_pipeline(LLM_MODEL_PATH)
    agent = make_agent(pipe, DEFAULT_REACT_JSON_SYSTEM_PROMPT)

    chat_list = []
    user_input = input("USER: ")
    new_chat = True
    while user_input != "END":
        chat_list = main(user_input, chat_list, agent, new_chat)
        print("ASSISTANT:")
        assistant_content = chat_list[-1]["content"]
        for char in assistant_content:
            print(char, end='', flush=True)
            time.sleep(0.02)
        print()
        new_chat = False
        user_input = input("\nUSER: ")

    print("CHAT HISTORY:")
    print(chat_list)
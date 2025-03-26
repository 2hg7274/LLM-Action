import os
import json
import warnings
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import get_json_schema
from functions import get_current_weather, get_location_nickname
from custom_system_prompt import TOOL_CALL_SYSTEM_PROMPT
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
warnings.filterwarnings(action='ignore')



def make_function_list():
    function_list = []
    get_current_weather_schema = get_json_schema(get_current_weather)
    get_location_nickname_schema = get_json_schema(get_location_nickname)

    function_list.append(get_current_weather_schema['function'])
    function_list.append(get_location_nickname_schema['function'])

    available_functions = {
        "get_current_weather": get_current_weather,
        "get_location_nickname": get_location_nickname
    }

    return function_list, available_functions


def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model



def gen(model, tokenizer, chat_list):
    text = tokenizer.apply_chat_template(
        chat_list,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    
    model_inputs = text.to(model.device)


    generation_config = GenerationConfig(
        max_new_tokens=4096,
        temperature=0.8,
        top_p=0.8,
        top_k=100,
        do_sample=True
    )

    generated_ids = model.generate(
        **model_inputs,
        generation_config=generation_config
    )
    generated_ids =[
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    chat_list.append({"role":"assistant", "content":response})
    return chat_list



def main(user_input, avail_functions, chat_list, model, tokenizer, flag=False):
    if not flag:
        chat_list.append({"role":"user", "content":user_input})
    
    chat_list = gen(model, tokenizer, chat_list)

    # print("[DEBUG]: ", chat_list[-1])

    tool_call_tag = "<|tool_call|>"
    response_message = chat_list[-1]["content"]

    tool_call_index = response_message.find(tool_call_tag)
    if tool_call_index != -1:
        print("Tool Calling!!!")
        response_message = response_message[tool_call_index+len(tool_call_tag):].strip()
        
        try:
            tool_call = json.loads(response_message)
        except:
            print(response_message)

        function_name = tool_call["name"]
        function_to_call = avail_functions[function_name]
        function_args = tool_call["arguments"]
        function_response = function_to_call(**function_args)

        chat_list.append({"role":"tool_response", "content":function_response})

        chat_list = main("", avail_functions, chat_list, model, tokenizer, flag=True)

    return chat_list

   



if __name__=="__main__":
    functions, available_functions = make_function_list()
    
    PATH = "/media/user/datadisk2/LLM_models/Phi-4-mini-instruct"
    tokenizer, model = load_model(PATH)


    chat_list = [
        {"role":"system", "content":TOOL_CALL_SYSTEM_PROMPT, "tools": str(functions)}
    ]
    
    user_input = input("USER: ")
    while user_input !=  "END":
        chat_list = main(user_input, available_functions, chat_list, model, tokenizer)
        print("ASSISTANT:")
        assistant_content = chat_list[-1]["content"]
        for chat in assistant_content:
            print(chat, end='', flush=True)
            time.sleep(0.02)
        print()
        user_input = input("\nUSER: ")

    print("CHAT HISTORY:")
    print(chat_list)
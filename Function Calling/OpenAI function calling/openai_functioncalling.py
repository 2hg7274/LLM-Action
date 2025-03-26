import json
import time
from openai import OpenAI
from transformers.utils import get_json_schema
from functions import get_current_weather, get_location_nickname
from configs import OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

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


def main(user_input, functions, avail_functions, chat_list):
    chat_list.append({"role":"user", "content":user_input})
    response = client.chat.completions.create(
        model = "gpt-4o-mini-2024-07-18",
        messages = chat_list,
        functions = functions,
        function_call = "auto"
    )

    response_message = response.choices[0].message

    if response_message.function_call:
        function_name = response_message.function_call.name
        function_to_call = avail_functions[function_name]
        function_args = json.loads(response_message.function_call.arguments)
        function_response = function_to_call(**function_args)
        
        chat_list.append(response_message)
        chat_list.append(
            {"role":"function", "name":function_name, "content":function_response}
        )

        after_function_response = client.chat.completions.create(
            model = "gpt-4o-mini-2024-07-18",
            messages=chat_list
        )
        chat_list.append(
            {"role":"assistant", "content":after_function_response.choices[0].message.content}
        )
    else:
        chat_list.append(
            {"role":"assistant", "content":response_message.content}
        )

    return chat_list



if __name__=="__main__":
    functions, available_functions = make_function_list()
    
    chat_list = []
    user_input = input("USER: ")
    while user_input != "END":
        chat_list = main(user_input, functions, available_functions, chat_list)
        print("ASSISTANT:")
        assistant_content = chat_list[-1]["content"]
        for chat in assistant_content:
            print(chat, end='', flush=True)
            time.sleep(0.02)
        print()
        user_input = input("\nUSER: ")

    print("CHAT HISTORY:")
    print(chat_list)
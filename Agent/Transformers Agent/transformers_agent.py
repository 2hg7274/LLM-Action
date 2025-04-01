import os
import time
import ast
import warnings
from transformers.agents.llm_engine import TransformersEngine
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    ReactJsonAgent
)
from custom_prompt import DEFAULT_REACT_JSON_SYSTEM_PROMPT
from tools import WebSearchTool, TopProcessesByMemoryTool, ProcessKillerTool, KakaoMessageTool


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings(action='ignore')

#####################################
# Agent Functions (Your original code)
#####################################

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
    web_search_tool = WebSearchTool()
    execute_tool = TopProcessesByMemoryTool()
    process_killer_tool = ProcessKillerTool()
    message_tool = KakaoMessageTool()
    return [web_search_tool, execute_tool, process_killer_tool, message_tool]

def make_agent(pipe, system_prompt):
    llm_engine = TransformersEngine(pipeline=pipe)
    llm_engine.tokenizer = llm_engine.pipeline.tokenizer
    tools = make_tools()
    agent = ReactJsonAgent(
        tools=tools,
        llm_engine=llm_engine,
        max_iterations=10
        # system_prompt=system_prompt
    )
    return agent

def main(user_input, chat_list, agent, chat_flag):
    if len(chat_list) == 0:
        chat_list.append({"role": "system", "content": agent.system_prompt})
    chat_list.append({"role": "user", "content": user_input})
    tokenized_user_input = agent.llm_engine.tokenizer.apply_chat_template(
        chat_list, tokenize=False, add_generation_prompt=True
    )
    if chat_flag:
        output = agent.run(tokenized_user_input)
    else:
        output = agent.run(tokenized_user_input, reset=False)
    chat_list.append({"role": "assistant", "content": output})
    return chat_list 


#####################################
# Alarm 입력 처리 함수
#####################################
def combine_alarm_predictions(alarm_list):
    """
    alarm 예측 리스트(딕셔너리 리스트)를 하나의 설명 텍스트로 변환합니다.
    """
    combined_texts = []
    for alarm in alarm_list:
        text = (f"Timestamp: {alarm.get('timestamp', 'N/A')}, CPU usage: {alarm.get('cpu_usage_percent', 'N/A')}%, "
                f"Memory usage: {alarm.get('memory_usage_percent', 'N/A')}%, Disk usage: {alarm.get('disk_usage_percent', 'N/A')}%, "
                f"Confidence: {alarm.get('confidence', 'N/A')}, Error occurred: {alarm.get('error_occur', 'N/A')}.")
        combined_texts.append(text)
    return " ".join(combined_texts)

def process_alarm_input(user_input):
    """
    만약 user_input이 alarm 예측 데이터(문자열 형태의 리스트)라면,
    이를 파싱하여 설명 텍스트로 변환합니다.
    """
    try:
        parsed = ast.literal_eval(user_input)
        if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
            return combine_alarm_predictions(parsed)
        elif isinstance(parsed, dict):
            return combine_alarm_predictions([parsed])
    except Exception as e:
        return user_input



#####################################
# Main 실행
#####################################
if __name__ == "__main__":
    from configs import LLM_MODEL_PATH
    from faiss_utils.utils import (
        create_dataset_from_files,
        load_embedding_model_and_tokenizer,
        compute_embeddings_for_dataset,
        build_faiss_index,
        search_similar_documents_faiss
    )
    # Agent를 위한 파이프라인 생성
    pipe = make_pipeline(LLM_MODEL_PATH)
    agent = make_agent(pipe, DEFAULT_REACT_JSON_SYSTEM_PROMPT)
    
    
    # 문서 데이터셋 생성
    file_paths = ["./documents/error_resolution.txt", "./documents/wifi_resolution.txt"]
    dataset = create_dataset_from_files(file_paths)
    
    # 임베딩 모델 로드 및 문서 임베딩 계산
    tokenizer_emb, model_emb = load_embedding_model_and_tokenizer()
    dataset = compute_embeddings_for_dataset(dataset, tokenizer_emb, model_emb)
    dataset = build_faiss_index(dataset)
    
    chat_list = []
    user_input = input("USER: ")
    new_chat = True
    while user_input != "END":
        # alarm 데이터 형태라면 파싱하여 설명 텍스트로 변환
        processed_input = process_alarm_input(str(user_input))
        
        # Faiss를 사용하여 유사 문서 검색 (이미 저장된 인덱스 파일을 로드)
        similar_context = search_similar_documents_faiss(processed_input, dataset, tokenizer_emb, model_emb, k=2)
        
        # 사용자 입력(또는 alarm 설명 텍스트)와 유사 문서 컨텍스트 결합
        combined_input = f"{processed_input}\n\nRelated context:\n{similar_context}"
        
        chat_list = main(combined_input, chat_list, agent, new_chat)
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
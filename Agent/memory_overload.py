from transformers import AutoModelForCausalLM


if __name__=="__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "/media/user/datadisk2/LLM_models/Qwen2.5-14B-Instruct-1M"
    )
    while True:
        user_input = input("TEST")
        
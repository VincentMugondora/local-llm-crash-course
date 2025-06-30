from typing import List
from ctransformers import AutoModelForCausalLM

# Load Llama2-7B-Chat model
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    model_file="llama-2-7b-chat.Q4_0.gguf"
)

# Define prompt using Llama2 chat format
def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answers. Answer with just the correct word, nothing more."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    print(f"Prompt created: {prompt}")
    return prompt

# Provide the question
question = "The name of the capital of Zimbabwe?"

# Generate prompt and pass it to the model
prompt = get_prompt(question)
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)

print()

# Provide the question
question = "And which is of the United States?"

# Generate prompt and pass it to the model
prompt = get_prompt(question)
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)

print()

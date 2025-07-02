from longchain_community.llms import CTransformers

llm = CTransformers(
    model="zoltanctoth/orca_mini_3B-GGUF",
    model_file="orca-mini-3b.q4_0.gguf",
    model_type="llama2",
    max_new_tokens=20,
)   

prompt = f"### System:\nYou are an AI assistant that gives helpful answers. You answer the question in a short and concise way.\n\n### User:\n{instruction}\n\n### Response:\n"

print(llm.invoke("Which is the capital of France?"))
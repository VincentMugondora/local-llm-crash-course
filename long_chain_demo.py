from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = CTransformers(
    model="zoltanctoth/orca_mini_3B-GGUF",
    model_file="orca-mini-3b.q4_0.gguf",
    model_type="llama2",
    max_new_tokens=20,
)   

prompt_template = """### System:\nYou are an AI assistant that gives helpful answers. You answer the question in a short and concise way.
\n\n### User:\n{instruction}\n\n### Response:\n"""

prompt = PromptTemplate(template=prompt_template, input_variables=["instruction"])

chain = LLMChain(llm = llm, prompt = prompt, verbose=True)
print(chain.invoke({"instruction": "Which is the capital of France?"}))
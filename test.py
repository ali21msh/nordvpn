from llama_cpp import Llama, ChatCompletionRequestUserMessage, ChatCompletionRequestSystemMessage

llm = Llama(
    model_path="./Meta-Llama-3-8B-Instruct.Q6_K.gguf",
    n_gpu_layers=1,  # Uncomment to use GPU acceleration
    n_threads=8,
    # seed=1337, # Uncomment to set a specific seed
    n_ctx=2048,  # Uncomment to increase the context window,
)
# output = llm(
#     "Q: Name the planets in the solar system? A: ",  # Prompt
#     max_tokens=None,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
#     # stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#     echo=True  # Echo the prompt back in the output
# )  # Generate a completion, can also call create_completion
sysmsg = ChatCompletionRequestSystemMessage(role="system", content="answer in persian")
message = ChatCompletionRequestUserMessage(content="ﺎﺴﺘﻗﻼﻟ ﺐﻬﺗﺮﻫ ﯼﺍ ﭖﺮﺴﭘﻮﻠﯿﺳ؟", role="user")
output = llm.create_chat_completion(messages=[sysmsg, message], stream=True)

content = ""
for chunk in output:
    delta = chunk['choices'][0]['delta']
    if 'role' in delta:
        print("### " + delta['role'], end=':')
    elif 'content' in delta:
        print(delta['content'], end='')

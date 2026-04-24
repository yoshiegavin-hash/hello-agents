from my_llm import MyLLM

# 实例化我们重写的客户端，并指定 ollama provider
llm = MyLLM(
    provider="ollama",
    model="llama3.2:1b",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 准备消息
messages =[{"role": "user", "content": "你好，请介绍一下你自己。"}]

# 发起调用，think方法都已从父类继承，无法重写
response_stream = llm.think(messages)
# 打印响应
print("ModelScope Response:")
for chunk in response_stream:
    # chunk在my_llm库中已经打印过一遍，这里只需要pass即可
    # print(chunk, end="", flush=True)
    pass
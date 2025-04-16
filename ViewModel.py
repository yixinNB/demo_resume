import os

import requests
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from loguru import logger

from vector_db import vector_db
from weather import get_weather

load_dotenv()
# ----------------------------
# 初始化 LangChain LLM
# ----------------------------
session = requests.Session()
llm = ChatOpenAI(
    temperature=0.6,
    openai_api_key=os.getenv("API_KEY"),
    # openai_api_base="https://api.bianxie.ai/v1",
    openai_api_base="http://127.0.0.1:11434/v1",
    # openai_api_base="http://127.0.0.1:11434",
    model="gpt-3.5-turbo",
).bind(response_format={"type": "json_object"})

# ----------------------------
# 创建代理
# ----------------------------
agent_executor = create_react_agent(llm, [get_weather])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    verbose=True,
)
#只有调用天气工具function输出json,其余情况输出均输出markdown.
sys_prompt = r"""User may give you some helpful context, after the context,there is a question. 判断用户意图是简历咨询还是天气查询：

- 若为简历：分析简历内容并回答与教育、经验、技能、匹配度等相关问题。
- 若为天气：若用户未提供天气信息，仅提供城市名，则输出 `{"arguments": "{\"location\":\"中文城市名\"}", "name": "get_weather"}`，不做其他输出；若已提供天气信息，则总结天气情况。

Function call enables communication between the system and locally running tools servers that provide additional tools and resources to extend your capabilities. You can output json to call the function.
如果已经调用过天气查询工具(对话中存在`<tool_response>`标记),则不得继续调用天气查询.
输出内容为中文。If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""


# 调用代理处理用户请求
def process_user_message(prompt: str):
    qa_result = qa_chain.invoke(prompt)
    logger.info(f"知识库查询结果{qa_result}")

    request_prompt = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just follow the system prompt, you can output json to call tool get weather information. don't try to make up an answer.\n\n---\n\n{qa_result}\n\nQuestion: {prompt}\n"""

    try:
        response = agent_executor.invoke({"messages": [("system", sys_prompt), ("user", request_prompt)]})
        assistant_message: str = response["messages"][-1].content
        assert isinstance(assistant_message, str)
        logger.debug(f"AI response {assistant_message}")
        logger.info("用户请求处理成功，回答：{}", assistant_message)
        return assistant_message
    except Exception as e:
        logger.exception("代理处理失败：{}", e)
        return f"处理请求时出错: {e}"

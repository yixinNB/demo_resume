import streamlit as st
from dotenv import load_dotenv

from ViewModel import process_user_message

# 加载环境变量
load_dotenv()

# 如果没有聊天记录，则初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("智能助手")
st.markdown("我可以帮你查询天气，也可以回答关于简历的问题。")

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 将用户消息存入聊天历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    assistant_r = process_user_message(prompt).replace("\\\\n","\n").replace("\\n","\n")
    st.session_state.messages.append({"role": "assistant", "content": assistant_r})
    with st.chat_message("assistant"):
        st.markdown(assistant_r)

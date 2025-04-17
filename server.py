import os
from typing import Optional, List

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 强制系统只能看到物理GPU 1（对应逻辑设备cuda:0）

import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import torch

# 初始化FastAPI应用
app = FastAPI()

# 设置Loguru日志记录
logger.add("logs/app.log", rotation="500 MB", retention="10 days", compression="zip")


# 定义请求和响应的模型
class Message(BaseModel):
    role: str
    content: Optional[str]


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str
    stream: bool
    temperature: float
    tools: Optional[List]=None


# 加载本地模型和分词器
model_name = "/home/yx/finetune/models/qwen-resume/14_4e-5xn/"  # 这里可以替换为你实际的模型名称
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 设备设置（如果有 GPU，优先使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 定义聊天推理函数
def generate_response(messages: list[Message]) -> str:
    logger.info(tokenizer.apply_chat_template(
        [msg for msg in messages if msg.content],
        tokenize=False,
        truncation=True,
        padding=True,
        add_generation_prompt=True  # 添加生成提示（例如 "<|im_start|>assistant"）
    ))

    # 将文本输入到模型中生成响应
    inputs = tokenizer.apply_chat_template(
        [msg for msg in messages if msg.content],
        return_tensors="pt",
        truncation=True,
        padding=True,
        add_generation_prompt=True  # 添加生成提示（例如 "<|im_start|>assistant"）
    ).to(device)
    input_length = inputs.shape[1]
    output = model.generate(inputs,max_new_tokens=500)

    # 解码输出
    response_text = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
    return response_text
    # return r"""{"arguments": "{\"location\":\"chengdu\"}", "name": "get_weather"}"""


# 定义OpenAI风格的聊天接口
@app.post("/v1/chat/completions")
async def chat(chat_request: ChatRequest):
    # 输出请求内容到控制台和日志文件
    # logger.info(f"Received chat request: {chat_request.dict()}")

    # 处理消息并生成回应
    response_text = generate_response(chat_request.messages)
    logger.info(f"Generated response: {response_text}")
    try:
        j=json.loads(response_text)
        if "name" in j:
            message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": j,
                            "id": "call_37Tc0c3BI15QTevWqvluneuw",
                            "type": "function"
                        }
                    ]
                }
        else:
            raise Exception("xxx")
    except:
        message={
                    "role": "assistant",
                    "content": response_text,
                }

    # 返回生成的回应
    return {
        "id": "chatcmpl-xxxxxxxxxx",  # 生成一个随机的ID（可以按需生成）
        "object": "chat.completion",
        "created": 1625551995,  # 你可以根据当前时间戳动态生成
        "model": "gpt-2",  # 返回模型的名称
        "choices": [
            {
                "message": message,
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }


# 运行FastAPI应用
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=11434)

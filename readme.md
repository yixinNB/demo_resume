# 原始需求

基于Qwen2.5 0.5B清洗 微调 不用蒸馏

传入简历

用function call的方式查询天气,写成一个模块

例如"今天成都的天气是什么":
大模型(要求使用微调后的模型)调度模块,模块调API,获得的数据返回给模型,模型输出给用户

**要求**:

使用名字或个人信息将信息抽取出来,需要微调的过程

# 🎯 需求拆解

- 问简历
    - 查询简历内容
    - 询问岗位匹配度
    - 我需要什么样的人,你有什么经验
- 问天气
    - xx地天气如何
    - 该穿什么衣服
    - 是否该防晒
    - 是否该带雨伞
    - 历史/预测 (暂不执行)

# 资料查询

function call https://python.langchain.com/docs/how_to/function_calling/#streaming

openai 服务端function call返回格式: 已通过抓包获取,见抓包文件

Base vs
Instruct https://www.byteplus.com/en/topic/417606?title=what-is-the-difference-between-qwen-2-5-coder-base-and-instruct

微调 https://zhuanlan.zhihu.com/p/673789772

# 实际执行

##  微调
- 随机划分为训练集和验证集
- 最终人工测试防止过拟合
- 双重检查确保EOS被正确配置
- LoRA 配置（如秩 `r`、缩放因子 `alpha`、目标模块等）
- 采用apply_chat_template使用模型内置模板训练
- 学习率设置在2e-5至20e-5之间,最终采用4e-5
- 每20steps使用验证集评估一次,连续两次eval loss增加则终止训练并加载最佳模型

基于Qwen base的模型`#12`表现较好,优于后续训练的Instruct模型.

## Agent

```
─────────────────────────────────────────────────────
  User Input              Process               Output
─────────────────────────────────────────────────────
Streamlit UI input ────>      ViewModel     ────> Streamlit UI output
                        │               ↑
                        │               │ (data returned)
                        ├───────────> [weather tool] 
                        │               ↑ 
                        │               │ (vector results)
                        ├───────────> [FAISS DB]
                        │               ↑ 
                        │               │
                        ├────────────────
```

- 先查询向量数据库, 适配"易鑫所在大学当地今天天气如何"这类问题

## 服务端

采用fastapi将transformers包装为OpenAI API格式,支持function call格式.

设置max_new_tokens防止无限生成.

# 遇到的问题

- gpt生成的langchain代码乱且无法运行
- 服务端支持function call
- 微调过拟合/幻觉
- system prompt过长
- api不支持中文,qwen中文转拼音差
- api返回长度超出模型输入的范围
- 数据集让gpt生成靠不住
- 当用户询问"湖州市"这种带"市"/"县"的时候无法查询
- 天气查询错误处理:让llm询问用户

# 系统文档

## 1. 系统概述

本系统是一个基于Streamlit开发的智能助手应用，集成了简历查询和天气查询两大核心功能。系统采用模块化设计，使用LangChain框架实现智能对话，并结合向量数据库进行简历信息检索。

## 2. 系统架构

### 2.1 核心组件

- **前端界面 (app.py)**
    - 基于Streamlit构建的Web应用界面
    - 实现用户交互和消息展示
    - 维护会话状态管理

- **向量数据库 (vector_db.py)**
    - 使用FAISS实现向量存储
    - 采用HuggingFace的多语言嵌入模型
    - 存储简历相关文档信息

- **业务逻辑层 (ViewModel.py)**
    - 集成LangChain框架
    - 实现智能代理系统
    - 处理用户意图识别和响应

- **天气服务 (weather.py)**
    - 对接Visual Crossing天气API
    - 实现天气数据获取和处理
    - 提供天气信息格式化输出

### 2.2 技术栈

- **前端框架**: Streamlit
- **AI框架**: LangChain
- **向量数据库**: FAISS
- **嵌入模型**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **天气API**: Visual Crossing
- **开发语言**: Python 3.x

## 3. 功能模块详解

### 3.1 用户界面模块

```
# app.py
- 初始化会话状态
- 实现聊天界面
- 处理用户输入
- 展示对话历史
```

### 3.2 向量数据库模块

```
# vector_db.py
- 文档预处理
- 向量化存储
- 相似度检索
- 支持多语言处理
```

### 3.3 业务逻辑模块

```
# ViewModel.py
- LLM模型配置
- 智能代理创建
- 意图识别
- 响应生成
```

### 3.4 天气服务模块

```
# weather.py
- 天气API调用
- 数据解析处理
- 天气信息格式化
- 错误处理机制
```

## 4. 数据流

1. 用户输入 -> 前端界面
2. 前端界面 -> 业务逻辑层
3. 业务逻辑层 -> 向量数据库/天气服务
4. 向量数据库/天气服务 -> 业务逻辑层
5. 业务逻辑层 -> 前端界面
6. 前端界面 -> 用户展示

## 5. 系统特点

- **多模态交互**: 支持自然语言交互
- **智能检索**: 基于向量数据库的语义搜索
- **模块化设计**: 各组件独立可扩展
- **实时响应**: 快速处理用户请求
- **多语言支持**: 支持中英文处理

## 6. 部署要求

- Python 3.x环境
- 必要的Python包依赖
- 环境变量配置（API密钥等）
- 网络连接（用于天气API调用）

## 7. 注意事项

- 确保环境变量正确配置
- 注意API调用频率限制
- 定期更新向量数据库
- 监控系统性能 

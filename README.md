
#  Dify 自定义工具 


## IntelliQ 意图识别和参数提取
IntelliQ 是一个基于大型语言模型（LLM）的多轮问答系统。该系统结合了先进的意图识别和词槽填充（Slot Filling）技术，致力于提升对话系统的理解深度和响应精确度。


### 特性
1. 多轮对话管理：能够处理复杂的对话场景，支持连续多轮交互。
2. 意图识别：准确判定用户输入的意图，支持自定义意图扩展。
3. 词槽填充：动态识别并填充关键信息（如时间、地点、对象等）。
4. 接口槽技术：直接与外部APIs对接，实现数据的实时获取和处理。
5. 自适应学习：不断学习用户交互，优化回答准确性和响应速度。
6. 易于集成：提供了详细的API文档，支持多种编程语言和平台集成。




## 安装和使用

确保您已安装 git、python3。然后执行以下步骤：
```
# 安装步骤
git clone https://github.com/answerlink/IntelliQ.git
cd IntelliQ
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 修改配置
配置项在 config/__init__.py
GPT_URL: 可修改为OpenAI的代理地址
API_KEY: 修改为ChatGPT的ApiKey

# 启动
python app.py

# 可视化调试可以浏览器打开 demo/user_input.html 或 127.0.0.1:5000
```




## 版本更新

v0.1 2025-03-18 首次更新；流程设计

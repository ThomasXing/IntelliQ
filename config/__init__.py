print("Config package initialized.")

DEBUG = True

# MODEL ------------------------------------------------------------------------

# 模型支持OpenAI规范接口
GPT_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
MODEL = 'qwen-plus'
API_KEY = 'sk-932e5157024a4cb1baf709dc6005a01c'
SYSTEM_PROMPT = 'You are a helpful assistant.'

# MODEL ------------------------------------------------------------------------

# CONFIGURATION ------------------------------------------------------------------------

# 意图相关性判断阈值0-1
RELATED_INTENT_THRESHOLD = 0.8

# CONFIGURATION ------------------------------------------------------------------------

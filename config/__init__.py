print("Config package initialized.")

DEBUG = True

# MODEL ------------------------------------------------------------------------

# 模型支持OpenAI规范接口
GPT_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
API_KEY = 'sk-932e5157024a4cb1baf709dc6005a01c'
# MODEL = 'qwen-max' 
MODEL = 'qwen-plus'
# MODEL = 'qwen2.5-14b-instruct'
SYSTEM_PROMPT = 'You are a helpful assistant.'
# GPT_URL = 'https://pro.aiskt.com/v1/chat/completions'
# API_KEY = 'sk-FebklebUyanCKfKS55C70eBeD9Ec4318A115A5A3FeDfAe6c'

# MODEL ------------------------------------------------------------------------

# CONFIGURATION ------------------------------------------------------------------------

# 意图相关性判断阈值0-1
RELATED_INTENT_THRESHOLD = 0.8

# CONFIGURATION ------------------------------------------------------------------------

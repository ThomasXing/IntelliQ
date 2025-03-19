# encoding=utf-8
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.chatbot_model import ChatbotModel
from utils.app_init import before_init
from utils.helpers import load_all_scene_configs
import json
import asyncio

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 实例化ChatbotModel
chatbot_model = ChatbotModel(load_all_scene_configs())

# 定义请求模型
class QuestionRequest(BaseModel):
    question: str
    session_id: str

from fastapi.responses import StreamingResponse
from functools import lru_cache

def get_cached_response(question: str, session_id: str):
    return chatbot_model.process_multi_question(question, session_id)

@app.post('/multi_question')
async def api_multi_question(request: QuestionRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="No question provided")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="No session_id provided")
    
    print('session_id:', request.session_id)
    
    # 获取响应内容
    response = get_cached_response(request.question, request.session_id)
    
    async def generate_response():
        # 使用更高效的分块处理
        chunk_size = 32  # 每次发送32个字符
        total_length = len(response)
        position = 0
        
        while position < total_length:
            # 获取当前块
            chunk = response[position:position + chunk_size]
            position += chunk_size
            
            # 动态调整延迟
            delay = 0.02 if len(chunk.strip()) > 0 else 0.01  # 有内容的块延迟稍长
            
            yield chunk
            await asyncio.sleep(delay)
    
    return StreamingResponse(generate_response(), media_type='text/plain')

@app.get('/')
async def index():
    return FileResponse('./demo/user_input.html')

if __name__ == '__main__':
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5100, help='服务器端口号')
    args = parser.parse_args()
    
    before_init()
    uvicorn.run(app, host='0.0.0.0', port=args.port)
# encoding=utf-8
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.chatbot_model import ChatbotModel
from utils.app_init import before_init
from utils.helpers import load_all_scene_configs
import json

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

@app.post('/multi_question')
async def api_multi_question(request: QuestionRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="No question provided")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="No session_id provided")
    
    print('session_id:', request.session_id)
    
    async def generate_stream():
        async for message in chatbot_model.process_multi_question(request.question, request.session_id):
            yield f"data: {json.dumps({'answer': message}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

@app.get('/')
async def index():
    return FileResponse('./demo/user_input.html')

if __name__ == '__main__':
    before_init()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
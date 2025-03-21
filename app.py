# encoding=utf-8
from models.chatbot_model_langchain import ChatbotModelLangChain
from utils.app_init import before_init
from utils.helpers import load_all_scene_configs
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

# 实例化ChatbotModel - 使用LangChain实现
chatbot_model = ChatbotModelLangChain(load_all_scene_configs())


@app.route('/multi_question', methods=['GET'])
async def api_multi_question():
    question = request.args.get('question')
    session_id = request.args.get('session_id', 'default_session')
    stream = request.args.get('stream', 'true').lower() == 'true'  # 默认启用流式响应
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # 使用异步方法处理请求
    response = await chatbot_model.process_multi_question(question, session_id)
    
    # 如果不使用流式响应，直接返回完整结果
    if not stream:
        return jsonify({"response": response})
    
    # SSE响应
    def generate():
        chunk_size = 32  # 每次发送32个字符
        total_length = len(response)
        position = 0
        
        # SSE需要以data:开头的行
        yield "data: {}\n\n".format("开始传输响应...")
        
        while position < total_length:
            # 获取当前块
            chunk = response[position:position + chunk_size]
            position += chunk_size
            
            # 动态调整延迟
            delay = 0.02 if len(chunk.strip()) > 0 else 0.01
            
            # 格式化为SSE事件
            yield "data: {}\n\n".format(chunk)
            time.sleep(delay)
            
        # 发送结束信号
        yield "data: [DONE]\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/', methods=['GET'])
def index():
    return send_file('./demo/user_input.html')


if __name__ == '__main__':
    before_init()
    app.run(port=5100, debug=True, threaded=True)

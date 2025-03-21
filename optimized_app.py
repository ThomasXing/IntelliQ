# encoding=utf-8
import json
from models.optimized_chatbot_model import OptimizedChatbotModel
from utils.app_init import before_init
from utils.helpers import load_all_scene_configs
from utils.json_serializer import LangChainJSONEncoder
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import time
import asyncio

app = Flask(__name__)
CORS(app)

# 配置Flask应用使用自定义JSON编码器
app.json_encoder = LangChainJSONEncoder

# 实例化优化版ChatbotModel
# 加载场景模板
with open('scene_config/scene_templates.json') as f:
    scene_templates = json.load(f)

chatbot_model = OptimizedChatbotModel(scene_templates=load_all_scene_configs(),
    max_cache_size=1000,
    cache_ttl=3600
)


@app.route('/multi_question', methods=['GET'])
async def api_multi_question():
    question = request.args.get('question')
    session_id = request.args.get('session_id', 'default_session')
    stream = request.args.get('stream', 'true').lower() == 'true'  # 默认启用流式响应
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # 使用异步方法处理请求
    if stream:
        # 流式响应模式
        def generate():
            # 创建一个事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # SSE需要以data:开头的行
            yield "data: {}\n\n".format("开始传输响应...")
            
            # 处理流式响应
            async def process_stream():
                try:
                    # 获取异步生成器
                    response_generator = await chatbot_model.aprocess_multi_question(
                        question, session_id, streaming=True
                    )
                    
                    # 现在response_generator是一个真正的异步生成器，可以使用async for
                    async for token in response_generator:
                        # 处理可能的AIMessage对象
                        if hasattr(token, 'content'):
                            token = token.content
                        
                        # 确保token是字符串
                        if not isinstance(token, str):
                            token = str(token)
                            
                        # 动态调整延迟
                        delay = 0.02 if len(token.strip()) > 0 else 0.01
                        # 格式化为SSE事件 - 确保token是字符串
                        token_str = str(token)
                        yield "data: {}\n\n".format(token_str)
                        await asyncio.sleep(delay)  # 使用异步sleep替代同步sleep
                except Exception as e:
                    print(f"流式响应错误: {e}")
                finally:
                    # 发送结束信号
                    yield "data: [DONE]\n\n"
            
            # 运行异步函数并返回结果
            try:
                # 使用run_until_complete来运行协程
                for chunk in loop.run_until_complete(collect_stream_results(process_stream())):
                    yield chunk
            finally:
                # 清理事件循环
                loop.close()
        
        return Response(generate(), mimetype='text/event-stream')
    else:
        # 非流式响应模式
        response = await chatbot_model.aprocess_multi_question(question, session_id)
        return jsonify({"response": response})


# 辅助函数，用于收集异步生成器的结果
async def collect_stream_results(generator):
    results = []
    try:
        async for item in generator:
            try:
                # 处理可能的AIMessage对象
                if hasattr(item, 'content'):
                    item = item.content
                
                # 确保item是字符串
                if not isinstance(item, str):
                    item = str(item)
                
                results.append(item)
            except Exception as e:
                print(f"处理流式响应项错误: {e}")
                # 继续处理下一个项，不中断整个流程
                continue
    except Exception as e:
        print(f"收集流式响应结果错误: {e}")
    return results


@app.route('/', methods=['GET'])
def index():
    return send_file('./demo/user_input.html')


@app.route('/clear_session', methods=['POST'])
def clear_session():
    session_id = request.json.get('session_id')
    if not session_id:
        return jsonify({"error": "No session_id provided"}), 400
    
    chatbot_model.clear_session(session_id)
    return jsonify({"status": "success", "message": f"Session {session_id} cleared"})


@app.route('/session_data', methods=['GET'])
async def get_session_data():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "No session_id provided"}), 400
    
    session_data = await chatbot_model.aget_session_data(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    
    return jsonify(session_data)


if __name__ == '__main__':
    before_init()
    # 定期清理过期会话
    import threading
    def clean_expired_sessions():
        while True:
            time.sleep(300)  # 每5分钟清理一次
            chatbot_model.clear_expired_sessions()
            print("已清理过期会话")
    
    # 启动清理线程
    cleaning_thread = threading.Thread(target=clean_expired_sessions, daemon=True)
    cleaning_thread.start()
    
    # 启动Flask应用
    app.run(port=5100, debug=True, threaded=True)
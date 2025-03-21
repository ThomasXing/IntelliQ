# encoding=utf-8
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union, AsyncGenerator

from langchain_components.optimized_chains import OptimizedIntentProcessingChain
from models.optimized_session_manager import OptimizedSessionManager

class OptimizedChatbotModel:
    """优化的聊天机器人模型，使用LCEL和异步处理提高响应速度
    
    特点：
    1. 使用LangChain表达式语言(LCEL)重写链组件
    2. 实现高效的缓存机制减少重复LLM调用
    3. 优化异步处理流程，提高并行能力
    4. 重新设计会话状态管理，减少状态传递开销
    5. 支持流式响应
    """
    
    def __init__(self, scene_templates: dict, max_cache_size: int = 1000, cache_ttl: int = 3600):
        """初始化聊天机器人模型
        
        Args:
            scene_templates: 场景模板字典
            max_cache_size: 最大缓存条目数
            cache_ttl: 缓存生存时间（秒）
        """
        self.scene_templates: dict = scene_templates
        self.session_manager = OptimizedSessionManager(max_cache_size=max_cache_size)
        self.intent_chain = OptimizedIntentProcessingChain(
            scene_templates=scene_templates,
            max_cache_size=max_cache_size,
            cache_ttl=cache_ttl
        )
        self.response_callbacks: List[callable] = []
    
    def register_response_callback(self, callback: callable) -> None:
        """注册响应回调函数，用于流式响应
        
        Args:
            callback: 回调函数，接收token参数
        """
        self.response_callbacks.append(callback)
    
    def process_multi_question(self, user_input: str, session_id: str) -> str:
        """处理多轮问答（同步版本）
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            
        Returns:
            处理结果
        """
        # 获取或创建会话
        session = self.session_manager.get_session(session_id)
        if not session:
            session = self.session_manager.create_session(session_id)
        
        # 使用意图处理链处理用户输入
        result = self.intent_chain.invoke({"user_input": user_input, "session": session})
        
        # 更新会话
        self.session_manager.update_session(session_id, result["updated_session"])
        
        return result["response"]
    
    async def aprocess_multi_question(self, user_input: str, session_id: str, streaming: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """
        异步处理多轮问答，提高响应速度
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            streaming: 是否启用流式响应
            
        Returns:
            处理结果或异步生成器
        """
        # 获取或创建会话
        session = await self.session_manager.aget_session(session_id)
        if not session:
            session = await self.session_manager.acreate_session(session_id)
        
        # 使用异步意图处理链处理用户输入
        if streaming:
            # 流式响应模式 - 直接返回异步生成器而不是协程
            class AsyncResponseGenerator:
                def __init__(self, intent_chain, user_input, session, session_manager, session_id, callbacks):
                    self.intent_chain = intent_chain
                    self.user_input = user_input
                    self.session = session
                    self.session_manager = session_manager
                    self.session_id = session_id
                    self.callbacks = callbacks
                
                def __aiter__(self):
                    return self
                
                async def __anext__(self):
                    if not hasattr(self, 'stream_iterator'):
                        self.stream_iterator = self.intent_chain.astream({"user_input": self.user_input, "session": self.session}).__aiter__()
                        self.done = False
                    
                    try:
                        token = await self.stream_iterator.__anext__()
                        
                        # 处理可能的AIMessage对象
                        if hasattr(token, 'content'):
                            token = token.content
                        
                        # 确保token是字符串
                        if not isinstance(token, str):
                            token = str(token)
                            
                        # 调用所有注册的回调函数
                        for callback in self.callbacks:
                            callback(token)
                        return token
                    except StopAsyncIteration:
                        if not self.done:
                            # 更新会话（在流式响应完成后）
                            final_result = await self.intent_chain.ainvoke({"user_input": self.user_input, "session": self.session})
                            await self.session_manager.aupdate_session(self.session_id, final_result["updated_session"])
                            self.done = True
                        raise StopAsyncIteration
            
            return AsyncResponseGenerator(
                self.intent_chain,
                user_input,
                session,
                self.session_manager,
                session_id,
                self.response_callbacks
            )
        else:
            # 普通异步响应模式
            result = await self.intent_chain.ainvoke({"user_input": user_input, "session": session})
            
            # 更新会话
            await self.session_manager.aupdate_session(session_id, result["updated_session"])
            
            return result["response"]
    
    def get_session_data(self, session_id: str) -> Optional[Dict]:
        """获取会话数据
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话数据字典
        """
        return self.session_manager.get_session(session_id)
    
    async def aget_session_data(self, session_id: str) -> Optional[Dict]:
        """异步获取会话数据
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话数据字典
        """
        return await self.session_manager.aget_session(session_id)
    
    def clear_session(self, session_id: str) -> None:
        """清除会话
        
        Args:
            session_id: 会话ID
        """
        self.session_manager.clear_session(session_id)
    
    def clear_all_sessions(self) -> None:
        """清除所有会话"""
        for session_id in list(self.session_manager.cache.keys()):
            self.clear_session(session_id)
    
    def clear_expired_sessions(self) -> None:
        """清除过期会话"""
        self.session_manager.clear_expired_sessions()
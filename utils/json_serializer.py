# encoding=utf-8
import json
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult, Generation
from langchain_core.agents import AgentAction, AgentFinish

class LangChainJSONEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，用于处理LangChain对象的序列化
    支持AIMessage、HumanMessage、SystemMessage等LangChain消息类型
    以及Document、LLMResult、Generation等其他LangChain对象
    """
    
    def default(self, obj: Any) -> Any:
        # 处理LangChain消息对象
        if isinstance(obj, BaseMessage):
            # 将消息对象转换为字典
            result = {
                "type": obj.__class__.__name__,
                "content": obj.content
            }
            # 添加额外属性
            if hasattr(obj, 'additional_kwargs') and obj.additional_kwargs:
                result["additional_kwargs"] = obj.additional_kwargs
            return result
        
        # 处理Document对象
        elif isinstance(obj, Document):
            return {
                "page_content": obj.page_content,
                "metadata": obj.metadata
            }
        
        # 处理LLMResult对象
        elif isinstance(obj, LLMResult):
            return {
                "generations": obj.generations,
                "llm_output": obj.llm_output
            }
        
        # 处理Generation对象
        elif isinstance(obj, Generation):
            return {
                "text": obj.text,
                "generation_info": obj.generation_info
            }
        
        # 处理AgentAction对象
        elif isinstance(obj, AgentAction):
            return {
                "tool": obj.tool,
                "tool_input": obj.tool_input,
                "log": obj.log
            }
        
        # 处理AgentFinish对象
        elif isinstance(obj, AgentFinish):
            return {
                "return_values": obj.return_values,
                "log": obj.log
            }
        
        # 处理可能的异步生成器
        elif hasattr(obj, '__aiter__') and callable(getattr(obj, '__aiter__')):
            return str(obj)
            
        # 处理其他类型的LangChain对象
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        
        # 使用默认的JSON编码器处理其他类型
        try:
            return super().default(obj)
        except TypeError:
            # 如果无法序列化，则转换为字符串
            return str(obj)


def serialize_message(message: BaseMessage) -> Dict:
    """
    将LangChain消息对象序列化为字典
    
    Args:
        message: LangChain消息对象
        
    Returns:
        序列化后的字典
    """
    if isinstance(message, AIMessage):
        return {"type": "ai", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"type": "human", "content": message.content}
    elif isinstance(message, SystemMessage):
        return {"type": "system", "content": message.content}
    else:
        return {"type": "unknown", "content": str(message)}


def deserialize_message(data: Dict) -> BaseMessage:
    """
    将字典反序列化为LangChain消息对象
    
    Args:
        data: 序列化的字典
        
    Returns:
        LangChain消息对象
    """
    msg_type = data.get("type", "")
    content = data.get("content", "")
    
    if msg_type == "ai":
        return AIMessage(content=content)
    elif msg_type == "human":
        return HumanMessage(content=content)
    elif msg_type == "system":
        return SystemMessage(content=content)
    else:
        return HumanMessage(content=content)  # 默认为人类消息
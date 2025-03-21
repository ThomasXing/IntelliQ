# encoding=utf-8
from typing import Dict, List, Any, Optional, Callable, Union, AsyncGenerator
import asyncio
import time
from functools import lru_cache
from collections import OrderedDict
from langchain.chains import create_extraction_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.runnables.base import Runnable
from langchain_openai import ChatOpenAI
import config

from langchain_components.prompts import (
    intent_relation_template,
    intent_recognition_template,
    slot_update_template,
    slot_query_template
)
from utils.data_format_utils import extract_continuous_digits, extract_float
from config import RELATED_INTENT_THRESHOLD
from utils.date_utils import get_current_date
from utils.helpers import extract_json_from_string, get_raw_slot, update_slot, is_slot_fully_filled, format_name_value_for_logging, get_slot_update_json, get_slot_query_user_json
from .workflow import Workflow

# 添加LRUCache类实现
class LRUCache:
    """简单的LRU缓存实现"""
    
    def __init__(self, max_size=1000, ttl=3600):
        """初始化LRU缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl: 缓存生存时间（秒）
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.timestamps = {}
    
    def get(self, key):
        """获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或已过期则返回None
        """
        if key not in self.cache:
            return None
        
        # 检查是否过期
        if time.time() - self.timestamps.get(key, 0) > self.ttl:
            self.delete(key)
            return None
        
        # 更新访问顺序
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def set(self, key, value):
        """设置缓存项
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            # 如果缓存已满，删除最久未使用的项
            if len(self.cache) >= self.max_size:
                oldest = next(iter(self.cache))
                self.delete(oldest)
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def delete(self, key):
        """删除缓存项
        
        Args:
            key: 缓存键
        """
        if key in self.cache:
            del self.cache[key]
            if key in self.timestamps:
                del self.timestamps[key]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.timestamps.clear()

# 初始化LLM，使用工厂函数以便于配置和测试
def create_llm(temperature: float = 0.7, streaming: bool = False, callbacks: List = None):
    """创建LLM实例的工厂函数
    
    Args:
        temperature: 温度参数，控制响应的随机性
        streaming: 是否启用流式响应
        callbacks: 回调处理器列表
        
    Returns:
        ChatOpenAI实例
    """
    return ChatOpenAI(
        model_name=config.MODEL,
        temperature=temperature,
        streaming=streaming,
        callbacks=callbacks,
        api_key=config.API_KEY,
        base_url=config.GPT_URL
    )

# 默认LLM实例
llm = create_llm()
workflow = Workflow()

# 使用LCEL创建可复用的链组件
intent_relation_chain = (
    ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(intent_relation_template.template)
    ])
    | llm
    | RunnableLambda(extract_float)
)

intent_recognition_chain = (
    ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(intent_recognition_template.template)
    ])
    | llm
    | RunnableLambda(extract_continuous_digits)
)

slot_update_chain = (
    ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(slot_update_template.template)
    ])
    | llm
    | RunnableLambda(extract_json_from_string)
)

slot_query_chain = (
    ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(slot_query_template.template)
    ])
    | llm
)

def intent_relation_func(inputs: Dict[str, Any]) -> float:
    """意图关联性判断函数"""
    prompt = intent_relation_template.format(
        scene_description=inputs["scene_description"],
        user_input=inputs["user_input"]
    )
    response = llm.invoke(prompt)
    return extract_float(response.content)

async def aintent_recognition_chain(inputs: Dict[str, Any]) -> List[str]:
    """异步版意图识别链"""
    prompt = intent_recognition_template.format(
        options_prompt=inputs["options_prompt"],
        user_input=inputs["user_input"]
    )
    response = await llm.ainvoke(prompt)
    return extract_continuous_digits(response)

# 优化的意图处理链
class OptimizedIntentProcessingChain:
    """优化的意图处理链，使用LCEL和异步处理提高响应速度"""
    
    def __init__(self, scene_templates: Dict, max_cache_size: int = 1000, cache_ttl: int = 3600):
        """初始化意图处理链
        
        Args:
            scene_templates: 场景模板字典
            max_cache_size: 最大缓存条目数
            cache_ttl: 缓存生存时间（秒）
        """
        self.scene_templates = scene_templates
        self.cache = LRUCache(max_size=max_cache_size, ttl=cache_ttl)
        
        # 使用LCEL创建主处理链
        # 创建同步和异步处理链
        # self._create_sync_chains()
        self._create_async_chains()

    def _create_sync_chains(self):
        """创建同步处理链"""
        self.main_chain = (
            RunnablePassthrough.assign(
                is_related=lambda x: self._is_related_to_last_intent(
                    x["session"].get('current_purpose', ''),
                    x["user_input"]
                )
            )
            | RunnableLambda(self._update_session_purpose)
            | RunnableLambda(self._aprocess_current_intent)
        )

    def _create_async_chains(self):
        """创建异步处理链"""
        self.aslot_update_chain = (
            ChatPromptTemplate.from_messages([
                HumanMessagePromptTemplate.from_template(slot_update_template.template)
            ])
            | llm
            | RunnableLambda(extract_json_from_string)
        )
        
        # 添加异步槽位查询链
        self.aslot_query_chain = (
            ChatPromptTemplate.from_messages([
                HumanMessagePromptTemplate.from_template(slot_query_template.template)
            ])
            | llm
        )
        
        self.async_chain = (
            RunnablePassthrough.assign(is_related=RunnableLambda(self._ais_related_to_last_intent))
            | RunnableLambda(self._update_session_purpose)
            | RunnableLambda(self._aprocess_current_intent)
        )
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """同步调用处理链
        
        Args:
            inputs: 输入数据，包含user_input和session
            
        Returns:
            处理结果，包含response和updated_session
        """
        return self.main_chain.invoke(inputs)
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """异步调用处理链
        
        Args:
            inputs: 输入数据，包含user_input和session
            
        Returns:
            处理结果，包含response和updated_session
        """
        user_input = inputs["user_input"]
        session = inputs["session"]
        
        # 检查当前输入是否与上一次的意图场景相关
        current_purpose = session.get('current_purpose', '')
        is_related = False
        
        if current_purpose:
            is_related = await self._ais_related_to_last_intent(current_purpose, user_input)
        
        if not is_related:
            # 不相关时，重新识别意图
            new_purpose = await self._arecognize_intent(user_input)
            if new_purpose:
                session['current_purpose'] = new_purpose
        
        # 处理当前意图
        if session.get('current_purpose') in self.scene_templates:
            response = await self._aprocess_intent(user_input, session)
        else:
            response = '未命中场景'
        
        return {"response": response, "updated_session": session}
    
    async def astream(self, inputs: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """异步流式调用处理链
        
        Args:
            inputs: 输入数据，包含user_input和session
            
        Returns:
            异步生成器，生成响应片段
        """
        # 获取输入数据
        user_input = inputs["user_input"]
        session = inputs["session"]
        
        # 检查当前输入是否与上一次的意图场景相关
        current_purpose = session.get('current_purpose', '')
        is_related = False
        
        if current_purpose:
            is_related = await self._ais_related_to_last_intent(current_purpose, user_input)
        
        if not is_related:
            # 不相关时，重新识别意图
            new_purpose = await self._arecognize_intent(user_input)
            if new_purpose:
                session['current_purpose'] = new_purpose
        
        # 处理当前意图
        if session.get('current_purpose') in self.scene_templates:
            # 使用异步方式处理意图并获取响应
            try:
                # 这里可能返回AIMessage对象或字符串
                response = await self._aprocess_intent(user_input, session)
                
                # 处理可能的AIMessage对象
                if hasattr(response, 'content'):
                    response = response.content
                    
                # 确保response是字符串
                if not isinstance(response, str):
                    response = str(response)
            except Exception as e:
                response = f'处理意图时出错: {str(e)}'
        else:
            response = '未命中场景'
        
        # 确保response是字符串
        if not isinstance(response, str):
            response = str(response)
        
        # 模拟流式输出
        for i in range(0, len(response), 4):
            token = response[i:i+4]
            # 处理可能的AIMessage对象
            if hasattr(token, 'content'):
                token = token.content
                
            # 确保token是字符串
            if not isinstance(token, str):
                token = str(token)
                
            yield token
            await asyncio.sleep(0.01)
    
    def _update_session_purpose(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """更新会话目的
        
        Args:
            inputs: 输入数据，包含user_input、session和is_related
            
        Returns:
            更新后的输入数据
        """
        user_input = inputs["user_input"]
        session = inputs["session"]
        is_related = inputs["is_related"]
        
        if not is_related:
            new_purpose = self._recognize_intent(user_input)
            if new_purpose:
                session["current_purpose"] = new_purpose
        
        return {"user_input": user_input, "session": session}
    
    async def _aprocess_current_intent(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """异步处理当前意图
        
        Args:
            inputs: 输入数据，包含user_input和session
            
        Returns:
            处理结果，包含response和updated_session
        """
        user_input = inputs["user_input"]
        session = inputs["session"]
        
        if session.get("current_purpose") in self.scene_templates:
            # 使用异步方式处理意图
            response = await self._aprocess_intent(user_input, session)
        else:
            response = "未命中场景"
        
        return {"response": response, "updated_session": session}

    async def _arespond_with_complete_data(self, scene_name: str, slot: List) -> str:
        """异步响应完整数据"""
        response = ""
        if slot:
            response = format_name_value_for_logging(slot) + f'\n正在请求{scene_name}API，请稍后……'
        else: 
            response = f'\n正在请求{scene_name}API，请稍后……'
        
        # 处理可能的AIMessage对象
        if hasattr(response, 'content'):
            response = response.content
            
        # 确保response是字符串
        if not isinstance(response, str):
            response = str(response)
            
        return response
    
    async def _is_related_to_last_intent(self, current_purpose: str, user_input: str) -> bool:
        """判断当前输入是否与上一次意图场景相关
        
        Args:
            current_purpose: 当前意图
            user_input: 用户输入
            
        Returns:
            是否相关
        """
        if not current_purpose or current_purpose not in self.scene_templates:
            return False
        
        # 使用缓存减少重复调用
        cache_key = f"relation:{current_purpose}:{user_input}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result > RELATED_INTENT_THRESHOLD
            
        scene_description = self.scene_templates[current_purpose]['description']
        result = await aintent_relation_chain({"scene_description": scene_description, "user_input": user_input})
        
        # 缓存结果
        self.cache.set(cache_key, result)
        return result > RELATED_INTENT_THRESHOLD
    
    def _recognize_intent(self, user_input: str) -> str:
        """识别用户意图
        
        Args:
            user_input: 用户输入
            
        Returns:
            识别出的意图，如果未识别出则返回None
        """
        # 使用缓存减少重复调用
        cache_key = f"intent:{user_input}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 根据场景模板生成选项
        purpose_options = {}
        purpose_description = {}
        index = 1
        
        for template_key, template_info in self.scene_templates.items():
            purpose_options[str(index)] = template_key
            purpose_description[str(index)] = template_info["description"]
            index += 1
            
        options_prompt = "\n".join([f"{key}. {value} - 请回复{key}" for key, value in purpose_description.items()])
        options_prompt += "\n0. 其他场景 - 请回复0"
        
        # 识别意图
        user_choices = intent_recognition_chain.invoke({"options_prompt": options_prompt, "user_input": user_input})
        
        # 根据用户选择获取对应场景
        result = None
        if user_choices and user_choices[0] != '0' and user_choices[0] in purpose_options:
            result = purpose_options[user_choices[0]]
        
        # 缓存结果
        self.cache.set(cache_key, result)
        return result
    
    async def _arecognize_intent(self, user_input: str) -> str:
        """异步识别用户意图
        
        Args:
            user_input: 用户输入
            
        Returns:
            识别出的意图，如果未识别出则返回None
        """
        # 使用缓存减少重复调用
        cache_key = f"intent:{user_input}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 根据场景模板生成选项
        purpose_options = {}
        purpose_description = {}
        index = 1
        
        for template_key, template_info in self.scene_templates.items():
            purpose_options[str(index)] = template_key
            purpose_description[str(index)] = template_info["description"]
            index += 1
            
        options_prompt = "\n".join([f"{key}. {value} - 请回复{key}" for key, value in purpose_description.items()])
        options_prompt += "\n0. 其他场景 - 请回复0"
        
        # 异步识别意图
        user_choices = await aintent_recognition_chain({"options_prompt": options_prompt, "user_input": user_input})
        
        # 根据用户选择获取对应场景
        result = None
        if user_choices and user_choices[0] != '0' and user_choices[0] in purpose_options:
            result = purpose_options[user_choices[0]]
        
        # 缓存结果
        self.cache.set(cache_key, result)
        return result
    
    async def _aprocess_intent(self, user_input: str, session: Dict) -> str:
        """异步处理特定意图进行参数提取"""
        current_purpose = session['current_purpose']
        scene_config = self.scene_templates[current_purpose]

        # 使用异步链处理槽位更新和查询
        slot_update_task = self.aslot_update_chain.ainvoke({
            "scene_name": scene_config["name"],
            "current_date": get_current_date(),
            "dynamic_example": scene_config.get('example', '{"name":"xx","value":"xx"}'),
            "slot_json": get_slot_update_json(session.get('slots', {})),
            "user_input": user_input
        })

        query_task = self.aslot_query_chain.ainvoke({
            "scene_name": scene_config["name"],
            "slot_json": get_slot_query_user_json(session.get('slots', {})),
            "user_input": user_input
        })

        updated_slots, query_result = await asyncio.gather(slot_update_task, query_task)
        
        # 处理可能的AIMessage对象
        if hasattr(updated_slots, 'content'):
            updated_slots = updated_slots.content
        if hasattr(query_result, 'content'):
            query_result = query_result.content
            
        # 确保updated_slots是字典
        if not isinstance(updated_slots, dict):
            try:
                updated_slots = extract_json_from_string(str(updated_slots))
            except:
                updated_slots = {}

        # 更新会话槽位
        session.setdefault('slots', {}).update(updated_slots)

        # 构造最终响应
        if is_slot_fully_filled(session['slots']):
            return await self._arespond_with_complete_data(scene_config["name"], session['slots'])
        return await self._aask_user_for_missing_data(scene_config["name"], session['slots'], user_input)


    
    async def _ais_related_to_last_intent(self, current_purpose: str, user_input: str) -> bool:
        """异步判断当前输入是否与上一次意图场景相关
        
        Args:
            current_purpose: 当前意图
            user_input: 用户输入
            
        Returns:
            是否相关
        """
        if not current_purpose or current_purpose not in self.scene_templates:
            return False
        
        # 使用缓存减少重复调用
        cache_key = f"relation:{current_purpose}:{user_input}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result > RELATED_INTENT_THRESHOLD
            
        scene_description = self.scene_templates[current_purpose]['description']
        result = await intent_relation_chain.ainvoke({"scene_description": scene_description, "user_input": user_input})
        
        # 缓存结果
        self.cache.set(cache_key, result)
        return result > RELATED_INTENT_THRESHOLD


    def _respond_with_complete_data(self, scene_name: str, slot: List) -> str:
        """当所有数据都准备好后的响应"""
        if slot:
            return format_name_value_for_logging(slot) + f'\n正在请求{scene_name}API，请稍后……'
        else: 
            return f'\n正在请求{scene_name}API，请稍后……'
    
    async def _ask_user_for_missing_data(self, scene_name: str, slot: List, user_input: str) -> str:
        """请求用户填写缺失的数据"""
        slot_json = get_slot_query_user_json(slot)
        return await self.aslot_query_chain.ainvoke({"scene_name": scene_name, "slot_json": slot_json, "user_input": user_input})
    
    async def _aask_user_for_missing_data(self, scene_name: str, slot: Dict, user_input: str) -> str:
        """异步请求用户填写缺失的数据"""
        # 获取第一个未填充的槽位
        missing_slot = get_raw_slot(slot)
        if not missing_slot:
            return await self._arespond_with_complete_data(scene_name, slot)
        
        # 使用缓存键
        cache_key = f"missing_data_{scene_name}_{missing_slot}_{user_input}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        slot_json = get_slot_query_user_json(slot)
        response = await self.aslot_query_chain.ainvoke({"scene_name": scene_name, "slot_json": slot_json, "user_input": user_input})
        
        # 处理可能的AIMessage对象
        if hasattr(response, 'content'):
            response = response.content
            
        # 确保response是字符串
        if not isinstance(response, str):
            response = str(response)
        
        # 缓存结果
        self.cache.set(cache_key, response)
        return response
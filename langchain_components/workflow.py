from typing import Dict, List, Optional, Union, Any
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.base import Chain
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.runnables.base import Runnable

class CustomCallbackHandler(BaseCallbackHandler):
    """自定义回调处理器，用于处理流式响应"""
    
    def __init__(self):
        self.response_text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """处理新的token"""
        self.response_text += token
        
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM响应结束时的处理"""
        pass

class Workflow:
    """基于LangChain的工作流程处理类"""
    
    def __init__(self, temperature: float = 0.7):
        """初始化Workflow实例
        
        Args:
            temperature: 温度参数，控制响应的随机性
        """
        self.temperature = temperature
        self.llm = self._create_llm()
        
    def _create_llm(self, streaming: bool = False) -> ChatOpenAI:
        """创建ChatOpenAI实例
        
        Args:
            streaming: 是否启用流式响应
            
        Returns:
            ChatOpenAI实例
        """
        import config
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else None
        return ChatOpenAI(
            model_name=config.MODEL,
            temperature=self.temperature,
            streaming=streaming,
            callbacks=callbacks,
            api_key=config.API_KEY,
            base_url=config.GPT_URL
        )
    
    def query_enterprise_knowledge_base(
        self,
        query: str,
        user_id: str = "default_user",
        conversation_id: str = "",
        system_prompt: str = "你是一个专业的知识库助手，请根据用户的问题提供准确的回答。",
        response_mode: str = "streaming",
        context: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], str]:
        """查询企业知识库
        
        Args:
            query: 用户查询文本
            user_id: 用户ID
            conversation_id: 对话ID
            system_prompt: 系统提示词
            response_mode: 响应模式，可以是"streaming"或"blocking"
            context: 额外的上下文信息
            
        Returns:
            处理后的响应文本或JSON数据
        """
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("{query}")
        ])
        
        # 使用LCEL创建对话链
        chain = (
            {"query": RunnablePassthrough(), "context": lambda x: context or {}}
            | prompt
            | self._create_llm(streaming=response_mode == "streaming")
        )
        
        try:
            if response_mode == "streaming":
                # 使用自定义回调处理器处理流式响应
                callback_handler = CustomCallbackHandler()
                response = chain.invoke(
                    query,
                    config={"callbacks": [callback_handler]}
                )
                return callback_handler.response_text
            else:
                # 直接返回完整响应
                return chain.invoke(query)
                
        except Exception as e:
            error_msg = f"查询知识库时出错: {str(e)}"
            print(error_msg)
            return error_msg
    
    def create_chat_chain(
        self,
        system_prompt: str,
        streaming: bool = False
    ) -> LLMChain:
        """创建聊天Chain
        
        Args:
            system_prompt: 系统提示词
            streaming: 是否启用流式响应
            
        Returns:
            配置好的LLMChain实例
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        return LLMChain(
            llm=self._create_llm(streaming=streaming),
            prompt=prompt,
            verbose=True
        )
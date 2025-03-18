# encoding=utf-8
import time
from typing import Dict, Optional

class SessionManager:
    def __init__(self, session_timeout: int = 1800):
        """初始化会话管理器
        
        Args:
            session_timeout (int): 会话超时时间（秒），默认30分钟
        """
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = session_timeout

    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话信息字典，如果会话不存在或已过期则返回None
        """
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        if self._is_session_expired(session):
            self.clear_session(session_id)
            return None
            
        session['last_access_time'] = time.time()
        return session

    def create_session(self, session_id: str) -> Dict:
        """创建新的会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            新创建的会话信息字典
        """
        session = {
            'id': session_id,
            'create_time': time.time(),
            'last_access_time': time.time(),
            'current_purpose': '',
            'processors': {}
        }
        self.sessions[session_id] = session
        return session

    def update_session(self, session_id: str, data: Dict) -> None:
        """更新会话信息
        
        Args:
            session_id: 会话ID
            data: 要更新的会话数据
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.update(data)
            session['last_access_time'] = time.time()

    def clear_session(self, session_id: str) -> None:
        """清除指定的会话
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

    def clear_expired_sessions(self) -> None:
        """清除所有过期的会话"""
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session['last_access_time'] > self.session_timeout
        ]
        for session_id in expired_sessions:
            self.clear_session(session_id)

    def _is_session_expired(self, session: Dict) -> bool:
        """检查会话是否已过期
        
        Args:
            session: 会话信息字典
            
        Returns:
            bool: 是否已过期
        """
        return time.time() - session['last_access_time'] > self.session_timeout
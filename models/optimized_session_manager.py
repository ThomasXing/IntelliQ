# encoding=utf-8
import time
import threading
import asyncio
import pickle
from collections import OrderedDict
from typing import Dict, Optional, Any
from collections import deque

class OptimizedSessionManager:
    """高性能会话管理器（双级缓存架构：内存+磁盘）"""
    
    def __init__(self, max_cache_size: int = 1000, cache_ttl: int = 3600):
        # 添加线程锁
        self._lock = threading.RLock()
        
        # 使用HighPerformanceCache实现内存缓存
        self.cache = HighPerformanceCache(max_size=max_cache_size, ttl=cache_ttl)
        
        # 磁盘缓存目录
        self.disk_cache_path = './session_cache'
        self.cache_ttl = cache_ttl

    def _clean_memory_cache(self):
        """触发缓存自动清理"""
        self.cache.clear_expired()
        self.cache._balance_cache()

    def get_session(self, session_id: str) -> Optional[Dict]:
        """双级缓存查询"""
        with self._lock:
            # 1. 高性能内存缓存
            if cached := self.cache.get(session_id):
                cached['last_access'] = time.time()  # 更新访问时间
                return cached
        
            # 2. 磁盘缓存
            if disk_data := self._load_disk(session_id):
                self._update_memory_cache(session_id, disk_data)
                return disk_data
        
        return None

    async def aget_session(self, session_id: str) -> Optional[Dict]:
        return await asyncio.to_thread(self.get_session, session_id)

    async def acreate_session(self, session_id: str) -> Dict:
        """异步创建新的会话"""
        return await asyncio.to_thread(self.create_session, session_id)

    def create_session(self, session_id: str) -> Dict:
        session = {
            'id': session_id,
            'created': time.time(),
            'data': {},
            'last_access': time.time()
        }
        self._update_memory_cache(session_id, session)
        return session

    def update_session(self, session_id: str, data: Dict) -> None:
        """更新会话信息
        
        Args:
            session_id: 会话ID
            data: 要更新的会话数据
        """
        with self._lock:
            if session_data := self.cache.get(session_id):
                session_data.update(data)
                session_data['last_access'] = time.time()
                self._update_memory_cache(session_id, session_data)

    async def aupdate_session(self, session_id: str, data: Dict) -> None:
        """异步更新会话"""
        await asyncio.to_thread(self.update_session, session_id, data)

    def _update_memory_cache(self, session_id: str, data: Dict):
        self.cache.set(session_id, data)

    def _load_disk(self, session_id: str) -> Optional[Dict]:
        # 示例磁盘加载逻辑
        return None

    def clear_expired_sessions(self):
        """清理所有过期会话"""
        with self._lock:
            current_time = time.time()
            
            # 清理内存缓存
            for session_id in list(self.cache.keys()):
                session = self.cache.get(session_id)
                if session and current_time - session.get('last_access', 0) > self.cache_ttl:
                    self.cache.delete(session_id)
    
    def _delete_session(self, session_id: str):
        with self._lock:
            self.cache.delete(session_id)
            
    def clear_session(self, session_id: str):
        """清除指定会话"""
        self._delete_session(session_id)

    def __del__(self):
        if hasattr(self, 'cleaner_running'):
            self.cleaner_running = False
            if hasattr(self, 'cleaner_thread') and self.cleaner_thread.is_alive():
                self.cleaner_thread.join()


class HighPerformanceCache:
    """高性能缓存实现（支持LRU/LFU策略）"""

    def __init__(self, max_size=1000, ttl=3600, segments=16, strategy='lru'):
        self.segment_count = segments
        self.segment_locks = [threading.RLock() for _ in range(segments)]
        self.segment_caches = [OrderedDict() for _ in range(segments)]
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl
        self._hits = 0
        self._misses = 0
        self.strategy = strategy
        self.access_counts = {}
        self._lock = threading.RLock()

    def _get_segment(self, key):
        return hash(key) % self.segment_count

    def get(self, key):
        seg = self._get_segment(key)
        with self.segment_locks[seg]:
            cache = self.segment_caches[seg]
            if key in cache:
                if time.time() - self.access_times[key] > self.ttl:
                    self._delete(key)
                    self._misses += 1
                    return None
                cache.move_to_end(key)
                self.access_times[key] = time.time()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                self._hits += 1
                return cache[key]
        self._misses += 1
        return None

    def set(self, key, value):
        seg = self._get_segment(key)
        with self.segment_locks[seg]:
            cache = self.segment_caches[seg]
            if key in cache:
                cache.move_to_end(key)
                self.access_counts[key] += 1
            else:
                cache[key] = value
                self.access_counts[key] = 1
            self.access_times[key] = time.time()
            self._balance_cache()

    def delete(self, key):
        seg = self._get_segment(key)
        with self.segment_locks[seg]:
            self._delete(key)

    def _delete(self, key):
        seg = self._get_segment(key)
        cache = self.segment_caches[seg]
        if key in cache:
            del cache[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.access_counts:
                del self.access_counts[key]

    def keys(self):
        """获取所有键"""
        all_keys = []
        for seg in range(self.segment_count):
            with self.segment_locks[seg]:
                all_keys.extend(list(self.segment_caches[seg].keys()))
        return all_keys

    def _balance_cache(self):
        total = sum(len(c) for c in self.segment_caches)
        while total > self.max_size:
            candidates = []
            for cache in self.segment_caches:
                if cache:
                    for key in cache:
                        seg = self._get_segment(key)
                        with self.segment_locks[seg]:
                            if self.strategy == 'lfu':
                                candidates.append( (key, self.access_counts.get(key,0)) )
                            else:
                                candidates.append( (key, self.access_times.get(key,0)) )
            
            # 根据策略排序淘汰候选
            reverse = self.strategy == 'lru'
            candidates.sort(key=lambda x: x[1], reverse=reverse)
            
            for key, _ in candidates[:total - self.max_size]:
                seg = self._get_segment(key)
                with self.segment_locks[seg]:
                    if key in self.segment_caches[seg]:
                        self._delete(key)
                        total -= 1
                if total <= self.max_size:
                    break

    def clear_expired(self):
        current_time = time.time()
        expired_keys = [
            k for k, t in self.access_times.items()
            if current_time - t > self.ttl
        ]
        for key in expired_keys:
            seg = self._get_segment(key)
            with self.segment_locks[seg]:
                if key in self.segment_caches[seg]:
                    self._delete(key)

    def clear(self):
        for seg in range(self.segment_count):
            with self.segment_locks[seg]:
                self.segment_caches[seg].clear()
        self.access_times.clear()
        self.access_counts.clear()

    @property
    def hit_rate(self):
        """获取当前缓存命中率"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def save_to_disk(self, file_path: str):
        """持久化缓存数据到磁盘"""
        data = {
            'segments': [dict(seg) for seg in self.segment_caches],
            'access_times': self.access_times,
            'access_counts': self.access_counts
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_from_disk(self, file_path: str):
        """从磁盘加载缓存数据"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        for i, seg_data in enumerate(data['segments']):
            self.segment_caches[i].update(seg_data)
            
        self.access_times.update(data['access_times'])
        self.access_counts.update(data['access_counts'])
        
        # 重建后执行缓存平衡
        self._balance_cache()
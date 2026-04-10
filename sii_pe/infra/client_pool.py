"""
多 API Key 负载均衡客户端池。

当并发数超过单个 key 的限制时，自动将请求分配到不同 key，
确保负载均衡和最大吞吐量。
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class ClientPool:
    """
    管理多个 AsyncOpenAI 客户端实例，按最低负载分配请求。

    用法：
        pool = ClientPool(["sk-key1", "sk-key2"], "https://api.deepseek.com")
        async with pool.get_client() as client:
            response = await client.chat.completions.create(...)
    """

    def __init__(
        self,
        api_keys: list[str],
        base_url: str = "https://api.deepseek.com",
        max_per_key: int = 10,
    ):
        if not api_keys:
            raise ValueError(
                "至少需要一个 API Key。"
                "请设置环境变量 SII_PE_API_KEYS（逗号分隔多个 key）"
            )

        self._clients: list[AsyncOpenAI] = []
        self._active_counts: list[int] = []
        self._max_per_key = max_per_key
        self._lock = asyncio.Lock()
        self._available = asyncio.Condition()

        for key in api_keys:
            client = AsyncOpenAI(api_key=key, base_url=base_url)
            self._clients.append(client)
            self._active_counts.append(0)

        logger.info(
            f"ClientPool 初始化完成: {len(api_keys)} 个 key, "
            f"每 key 最大并发 {max_per_key}, "
            f"总最大并发 {len(api_keys) * max_per_key}"
        )

    @property
    def total_capacity(self) -> int:
        return len(self._clients) * self._max_per_key

    @property
    def active_requests(self) -> int:
        return sum(self._active_counts)

    async def acquire(self) -> tuple[AsyncOpenAI, int]:
        """
        获取负载最低的 client。

        如果所有 client 都达到并发上限，阻塞等待直到有空闲。
        返回 (client, index) 元组。
        """
        async with self._available:
            while True:
                # 找到负载最低的 client
                min_idx = min(range(len(self._clients)), key=lambda i: self._active_counts[i])
                if self._active_counts[min_idx] < self._max_per_key:
                    self._active_counts[min_idx] += 1
                    return self._clients[min_idx], min_idx
                # 所有 client 都满载，等待
                await self._available.wait()

    async def release(self, index: int) -> None:
        """归还 client，减少其活跃计数。"""
        async with self._available:
            self._active_counts[index] = max(0, self._active_counts[index] - 1)
            self._available.notify_all()

    @asynccontextmanager
    async def get_client(self):
        """上下文管理器：自动获取和归还 client。"""
        client, index = await self.acquire()
        try:
            yield client
        finally:
            await self.release(index)

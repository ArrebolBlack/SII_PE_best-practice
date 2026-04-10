"""
异步 LLM 调用封装。

在 ClientPool 基础上添加重试、超时和结构化日志。
"""

import asyncio
import logging

from sii_pe.infra.client_pool import ClientPool

logger = logging.getLogger(__name__)


async def call_llm(
    pool: ClientPool,
    messages: list[dict],
    model: str = "deepseek-chat",
    temperature: float = 1.0,
    max_tokens: int = 8192,
    max_retries: int = 3,
    timeout: float = 120.0,
) -> str:
    """
    发送一次 LLM 请求，支持重试与超时。

    参数:
        pool: 客户端池
        messages: OpenAI Chat API 消息列表
        model: 模型名称
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        max_retries: 最大重试次数
        timeout: 单次请求超时（秒）

    返回:
        模型输出文本
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            async with pool.get_client() as client:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    timeout=timeout,
                )
                content = response.choices[0].message.content
                logger.debug(f"LLM 请求成功 (尝试 {attempt})")
                return content

        except asyncio.TimeoutError:
            last_error = TimeoutError(f"LLM 请求超时 ({timeout}s)")
            logger.warning(f"LLM 请求超时 (尝试 {attempt}/{max_retries})")
        except Exception as e:
            last_error = e
            logger.warning(f"LLM 请求失败 (尝试 {attempt}/{max_retries}): {e}")

        # 指数退避
        if attempt < max_retries:
            wait_time = 2 ** (attempt - 1)
            logger.info(f"等待 {wait_time}s 后重试...")
            await asyncio.sleep(wait_time)

    raise last_error

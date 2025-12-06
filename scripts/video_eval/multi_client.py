from openai import OpenAI, AzureOpenAI
import openai
import asyncio
import os
from typing import List, Dict, Any, Union
from loguru import logger
# 导入两种客户端类型
from openai import AsyncAzureOpenAI, AsyncOpenAI, RateLimitError, APIError

# --- 1. 定义一个能管理多种客户端的类 ---

class MultiClientManager:
    """
    一个可以轮询和管理多种 OpenAI 兼容客户端（如 Azure, 阿里云等）的管理器。
    它能根据配置自动分发请求到不同的客户端，并提供重试机制。
    """
    def __init__(self, client_configs: List[Dict[str, Any]]):
        """
        初始化管理器。
        :param client_configs: 一个配置字典的列表。每个字典代表一个客户端。
                               示例:
                               [
                                 {'type': 'azure', 'api_key': '...', 'azure_endpoint': '...', 'api_version': '...', 'model_deployment': 'my-gpt4'},
                                 {'type': 'openai', 'api_key': '...', 'base_url': '...', 'model_name': 'qwen-max'}
                               ]
        """
        if not client_configs:
            raise ValueError("客户端配置列表 (client_configs) 不能为空！")

        self._client_configs = client_configs
        self._clients_cache: Dict[int, Union[AsyncAzureOpenAI, AsyncOpenAI]] = {} # 缓存已创建的客户端

        # 用于轮询的索引和保证异步安全的锁
        self._current_index = 0
        self._lock = asyncio.Lock()

    def _create_client(self, config: Dict[str, Any]) -> Union[AsyncAzureOpenAI, AsyncOpenAI]:
        """
        客户端工厂：根据配置创建相应的客户端实例。
        """
        client_type = config.get("type")
        
        if client_type == "azure":
            logger.info(f"正在创建 Azure 客户端 (Endpoint: {config['azure_endpoint']})")
            return AsyncAzureOpenAI(
                api_key=config["api_key"],
                azure_endpoint=config["azure_endpoint"],
                api_version=config["api_version"],
            )
        elif client_type == "openai": # 适用于标准 OpenAI 或其他兼容服务
            logger.info(f"正在创建 OpenAI 兼容客户端 (Base URL: {config['base_url']})")
            return AsyncOpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"],
            )
        else:
            raise ValueError(f"不支持的客户端类型: {client_type}")

    async def _get_next_client_and_model(self) -> tuple[Union[AsyncAzureOpenAI, AsyncOpenAI], str]:
        """
        异步安全地获取下一个客户端实例及其对应的模型名称。
        """
        async with self._lock:
            # 获取当前索引和对应的配置
            index = self._current_index
            config = self._client_configs[index]
            
            # 更新索引，为下一次调用做准备
            self._current_index = (self._current_index + 1) % len(self._client_configs)
            
            # 使用缓存，如果客户端已创建则直接返回，否则创建并缓存
            if index not in self._clients_cache:
                self._clients_cache[index] = self._create_client(config)
            
            client = self._clients_cache[index]
            
            # 获取该客户端应该使用的模型名称
            # Azure 使用 'model_deployment', 其他使用 'model_name'
            model_name = config.get("model_deployment") or config.get("model_name")
            if not model_name:
                raise ValueError(f"配置索引 {index} 中缺少 'model_deployment' 或 'model_name'。")

            logger.debug(f"当前使用客户端 (索引 {index}), 类型: {config['type']}, 模型: {model_name}, Key: ...{config['api_key'][-4:]}")
            return client, model_name, index

    async def chat(self, messages: List[Dict[str, str]], max_retries: int = 5, **kwargs: Any) -> Any:
        """
        封装 chat.completions.create 方法，自动处理客户端轮询、模型切换和重试。
        
        :param messages: 消息列表。
        :param max_retries: 对于可重试错误的最大重试次数。
        :param kwargs: 其他传递给 chat.completions.create 的参数 (如 temperature, max_tokens)。
        :return: OpenAI API 的响应对象。
        """
        last_exception = None
        
        # 重试次数不应超过客户端的总数
        retries = min(len(self._client_configs), max_retries)

        for attempt in range(retries):
            try:
                # 1. 获取下一个客户端和它对应的模型名
                client, model_to_use, idx = await self._get_next_client_and_model()
                # 2. 发起 API 请求
                # 注意：我们将获取到的模型名传递给 create 方法
                chat_response = await client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    **kwargs
                )
            
            # 3. 如果成功，直接返回结果
                return chat_response

            except RateLimitError as e:
                logger.warning(f"请求遭遇速率限制 (尝试 {attempt + 1}/{retries})。正在使用下一个客户端重试...")
                last_exception = e
                await asyncio.sleep(1)
            except APIError as e:
                logger.warning(f"发生 API 错误 (尝试 {attempt + 1}/{retries}): {e}。正在使用下一个客户端重试...")
                last_exception = e
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"发生未知错误 (尝试 {attempt + 1}/{retries}): {e}。")
                last_exception = e
                raise e
        logger.error("所有客户端均尝试失败。")
        raise last_exception

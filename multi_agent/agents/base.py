"""LLM 适配层：一个极小的接口 + 两种实现。

* ``OpenAICompatibleClient``：任何兼容 ``/chat/completions`` 的服务（含本地 vLLM/Ollama 网关）；
* ``RuleBasedClient``：不调用任何模型，用确定性模板产出结构化工件，
  用于离线 demo、单元测试和「无网络环境」下的最小可运行路径。

环境变量：

    MA_LLM_BASE_URL   example: https://api.openai.com/v1
    MA_LLM_API_KEY
    MA_LLM_MODEL      example: gpt-4.1-mini
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        ...


class RuleBasedClient:
    """确定性兜底：不产生自由文本，只用于标记「本次未调用模型」。"""

    name = 'rule_based'

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        raise RuntimeError('RuleBasedClient 不生成文本；请使用 roles 里的启发式实现')


class OpenAICompatibleClient:
    name = 'openai_compatible'

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.calls = 0

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        import requests  # 延迟导入：离线路径不要求安装

        self.calls += 1
        resp = requests.post(
            f'{self.base_url}/chat/completions',
            headers={'Authorization': f'Bearer {self.api_key}',
                     'Content-Type': 'application/json'},
            json={
                'model': self.model,
                'temperature': 0.4,
                'max_tokens': max_tokens,
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']


def get_client() -> LLMClient:
    base = os.getenv('MA_LLM_BASE_URL', '').strip()
    key = os.getenv('MA_LLM_API_KEY', '').strip()
    model = os.getenv('MA_LLM_MODEL', '').strip()
    if base and model:
        return OpenAICompatibleClient(base, key, model)
    return RuleBasedClient()


def is_llm(client: LLMClient) -> bool:
    return client.name != 'rule_based'


_FENCE = re.compile(r'```(?:json)?\s*(.*?)```', re.S)


def parse_json(text: str) -> Dict[str, Any]:
    """尽力从模型输出里抠出第一个 JSON 对象。"""
    if not text:
        raise ValueError('空响应')
    m = _FENCE.search(text)
    candidate = m.group(1) if m else text
    start = candidate.find('{')
    end = candidate.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('响应里没有 JSON 对象')
    return json.loads(candidate[start:end + 1])


def env_llm_info() -> Optional[Dict[str, str]]:
    base = os.getenv('MA_LLM_BASE_URL', '').strip()
    if not base:
        return None
    return {'base_url': base, 'model': os.getenv('MA_LLM_MODEL', '')}

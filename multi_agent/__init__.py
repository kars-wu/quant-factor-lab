"""多 agent 因子研究系统。

模块：

* ``contracts``    候选/数据/执行/评估契约与哈希
* ``dsl``          受限因子表达式语言（算子与字段白名单）
* ``registry``     单写者 SQLite 台账（不可覆盖的产物与预算）
* ``datasets``     数据视图（合成 / 只读 SQLite）
* ``evaluation``   H 日 tranche 回测、指标、Gate0 不变量
* ``agents``       六种角色的 prompt 与实现（LLM 可选，离线有确定性兜底）
* ``orchestration``研究循环与预算执行
* ``evolution``    外层研究策略进化（配对评测）
"""

__version__ = '1.0.0'

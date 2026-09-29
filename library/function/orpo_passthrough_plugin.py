"""orpo_passthrough_plugin.py — axolotl 0.18 ORPO 용 passthrough 전략 플러그인.

문제:
    axolotl 0.18 의 RL 데이터 경로는 dataset `type` 을
    `axolotl.prompt_strategies.<rl>.<module>` 패키지 안에서 찾는다.
    DPO 패키지에는 passthrough.py (identity 변환) 가 있지만,
    ORPO 패키지에는 chat_template.py 하나뿐이고 그마저 로더 호출 규약과
    시그니처가 맞지 않아 사용할 수 없다.
    → rl: orpo + type: passthrough 조합이 "No module named
      'axolotl.prompt_strategies.orpo.passthrough'" 로 실패한다.

해결:
    axolotl plugins 메커니즘(config 의 `plugins:` 리스트)으로 이 플러그인을
    학습 시작 시 로드하여, 누락된 orpo.passthrough 모듈을 sys.modules 에
    등록한다. DPO 의 passthrough 와 동일한 identity transform 을 제공하므로
    {prompt, chosen[], rejected[]} 메시지 리스트 데이터를 그대로
    TRL ORPOTrainer 에 넘길 수 있다 (TRL 이 chat template 적용).

사용:
    config yaml 의 plugins 리스트에 추가:
        plugins:
          - orpo_passthrough_plugin.ORPOPassthroughPlugin
"""
from __future__ import annotations

import sys
import types

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

MODULE_NAME = "axolotl.prompt_strategies.orpo.passthrough"


def _register_passthrough_module() -> None:
    if MODULE_NAME in sys.modules:
        return

    # DPO 패키지의 공식 passthrough 구현(default: identity transform)을 재사용
    from axolotl.prompt_strategies.dpo.passthrough import default

    mod = types.ModuleType(MODULE_NAME)
    mod.default = default
    mod.__doc__ = "ORPO passthrough strategy (registered by orpo_passthrough_plugin)"
    sys.modules[MODULE_NAME] = mod
    LOG.info("[orpo_passthrough_plugin] registered %s", MODULE_NAME)


class ORPOPassthroughPlugin(BasePlugin):
    """rl: orpo 학습에서 dataset type: passthrough 를 사용 가능하게 하는 플러그인."""

    def register(self, cfg) -> None:
        _register_passthrough_module()

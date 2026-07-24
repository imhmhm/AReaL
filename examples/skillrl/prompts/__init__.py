"""Prompt templates for SkillRL envs (ported from SkillRL's ``environments/prompts/``).

Prompts live as a sibling of ``env_package/`` (mirroring the original
``agent_system/environments/prompts/`` layout, where ``prompts/`` is a sibling
of ``env_package/``), NOT inside ``env_package/``.
"""
from .alfworld import *  # noqa: F401,F403
from .search import *  # noqa: F401,F403
from .webshop import *  # noqa: F401,F403

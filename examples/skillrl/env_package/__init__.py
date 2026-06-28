"""Environment packages for SkillRL.

Each subpackage (e.g. ``search``) provides:
- ``build_*_envs``   : factory for the vectorized gym env
- ``*_projection``   : maps LLM text actions -> env actions + validity
- ``third_party``    : vendored env implementations (e.g. skyrl_gym)
"""

"""Compact dataset profile for LLM prompts: one line per column, under a hard token budget."""

from eda_engine.profile.compact import CompactProfile, build_compact_profile, fmt_number

__all__ = ["CompactProfile", "build_compact_profile", "fmt_number"]

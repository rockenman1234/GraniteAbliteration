#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025
"""
Granite 3.3 Abliteration Source Package

This package contains the core abliteration functionality for IBM Granite models.
"""

from .abliterate import abliterate_model, is_granite_model, fix_granite_config
from .direction_ablation import compute_refusal_direction, apply_direction_ablation, load_prompts_from_files

__all__ = [
    'abliterate_model',
    'is_granite_model', 
    'fix_granite_config',
    'compute_refusal_direction',
    'apply_direction_ablation',
    'load_prompts_from_files'
]

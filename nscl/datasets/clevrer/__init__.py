#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/29/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

from nscl.datasets.factory import register_dataset
from .definition import CLEVRERDefinition
from .definition import (
    build_clevrer_dataset,
    build_symbolic_clevrer_dataset,
    build_concept_retrieval_clevrer_dataset,
    build_concept_quantization_clevrer_dataset,
)

register_dataset(
    "clevrer",
    CLEVRERDefinition,
    builder=build_clevrer_dataset,
    symbolic_builder=build_symbolic_clevrer_dataset,
    concept_retrieval_builder=build_concept_retrieval_clevrer_dataset,
    concept_quantization_builder=build_concept_quantization_clevrer_dataset,
)

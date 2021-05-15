#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_translator.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/30/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Tools for translating programs into different formats.
"""

from copy import deepcopy

__all__ = ["clevrer_to_nsclseq"]

concept_frame = {
    "moving": [],
    "stationary": [],
}

concept_events = {
    "in": [],
    "out": [],
    "collision": [],
}

concept_ee = {
    "before": [],
    "after": [],
    "ancestor": [],
}

concept_order = {
    "order": [],
}

modules = {
    "objects": {"func": "objects", "nargs": 0},
    "events": {"func": "events", "nargs": 0},
    "unique": {"func": "unique", "nargs": 1},
    "count": {"func": "count", "nargs": 1},
    "exist": {"func": "exist", "nargs": 1},
    "negate": {"func": "negate", "nargs": 1},
    "belong_to": {"func": "belong_to", "nargs": 2},
    "filter_color": {"func": "filter_color", "nargs": 2},
    "filter_material": {"func": "filter_material", "nargs": 2},
    "filter_shape": {"func": "filter_shape", "nargs": 2},
    "filter_resting": {"func": "filter_resting", "nargs": 2},
    "filter_moving": {"func": "filter_moving", "nargs": 2},
    "filter_stationary": {"func": "filter_stationary", "nargs": 2},
    "filter_start": {"func": "filter_start", "nargs": 1},
    "filter_end": {"func": "filter_end", "nargs": 1},
    "filter_in": {"func": "filter_in", "nargs": 2},
    "filter_out": {"func": "filter_out", "nargs": 2},
    "filter_collision": {"func": "filter_collision", "nargs": 2},
    "filter_order": {"func": "filter_order", "nargs": 2},
    "filter_before": {"func": "filter_before", "nargs": 2},
    "filter_after": {"func": "filter_after", "nargs": 2},
    "query_color": {"func": "query_color", "nargs": 1},
    "query_material": {"func": "query_material", "nargs": 1},
    "query_shape": {"func": "query_shape", "nargs": 1},
    "query_direction": {"func": "query_direction", "nargs": 2},
    "query_frame": {"func": "query_frame", "nargs": 1},
    "query_object": {"func": "query_object", "nargs": 1},
    "query_collision_partner": {"func": "query_collision_partner", "nargs": 2},
    "get_col_partner": {"func": "query_collision_partner", "nargs": 2},  # ??
    "filter_ancestor": {"func": "filter_ancestor", "nargs": 2},
    "unseen_events": {"func": "unseen_events", "nargs": 0},
    "all_events": {"func": "all_events", "nargs": 0},
    "counterfact_events": {"func": "counterfact_events", "nargs": 1},
    "filter_counterfact": {"func": "filter_counterfact", "nargs": 2},
}


def get_clevr_pblock_op(block):
    """
    Return the operation of a CLEVR program block.
    """
    if "type" in block:
        return block["type"]
    assert "function" in block
    return block["function"]


def get_clevr_op_attribute(op):
    return op.split("_")[1]


def clevrer_to_clevr(clevrer_program):

    last_op_idx = -1
    clevr_program = list()
    exe_stack = []

    for m in clevrer_program:
        if m in ["<END>", "<NULL>"]:
            break
        if m not in ["<START>"]:
            if m not in modules:
                exe_stack.append(m)
            else:
                node = dict()
                node["inputs"] = []
                node["function"] = modules[m]["func"]
                argv = []
                for i in range(modules[m]["nargs"]):
                    if exe_stack:
                        pop = exe_stack.pop()

                        if isinstance(pop, dict):
                            pop = pop["id"]
                            node["inputs"].append(pop)
                        else:
                            argv.insert(0, pop)

                    else:
                        return "error"

                node["value_inputs"] = [*argv]
                last_op_idx += 1
                node["id"] = last_op_idx
                exe_stack.append(node)

                clevr_program.append(node)

    return clevr_program


def clevrer_to_nsclseq(clevrer_program):
    return clevr_to_nsclseq(clevrer_to_clevr(clevrer_program))


def clevr_to_nsclseq(clevr_program):

    nscl_program = list()
    mapping = dict()

    for block_id, block in enumerate(clevr_program):
        op = get_clevr_pblock_op(block)
        current = None
        if op == "objects":
            current = dict(op="objects")
        elif op == "events":
            current = dict(op="events")
        elif op.startswith("filter") and get_clevr_op_attribute(op) in concept_events:
            # concept = get_clevr_op_attribute(op)
            current = dict(op="filter_events")  # , concept=[concept])

        elif op.startswith("filter") and get_clevr_op_attribute(op) in concept_frame:
            concept = get_clevr_op_attribute(op)
            current = dict(op="filter_frame")  # , concept=[concept])

        elif op.startswith("filter") and get_clevr_op_attribute(op) in concept_ee:
            concept = get_clevr_op_attribute(op)
            current = dict(op="filter_ee")  # , concept=[concept])

        elif op.startswith("filter") and get_clevr_op_attribute(op) in concept_order:
            concept = get_clevr_op_attribute(op)
            current = dict(op="filter_order")  # , concept=[concept])

        elif op.startswith("filter"):
            concept = block["value_inputs"][0]
            last = nscl_program[mapping[block["inputs"][0]]]
            if last["op"] == "filter":
                last["concept"].append(concept)
            else:
                current = dict(op="filter", concept=[concept])
        elif op.startswith("relate"):
            concept = block["value_inputs"][0]
            current = dict(op="relate", relational_concept=[concept])
        elif op.startswith("same"):
            attribute = get_clevr_op_attribute(op)
            current = dict(op="relate_attribute_equal", attribute=attribute)
        elif op in ("intersect", "union"):
            current = dict(op=op)
        elif op == "unique":
            pass  # We will ignore the unique operations.
        else:
            if "query_collision_partner" in op:
                current = dict(op="query_collision_partner")
            elif op.startswith("query"):
                if block_id == len(clevr_program) - 1:
                    attribute = get_clevr_op_attribute(op)
                    current = dict(op="query", attribute=attribute)
            elif op.startswith("equal") and op != "equal_integer":
                attribute = get_clevr_op_attribute(op)
                current = dict(op="query_attribute_equal", attribute=attribute)
            elif op == "exist":
                current = dict(op="exist")

            elif op == "count":

                # if ...:
                #     op = "count_objects"
                # else:
                #     op = "count_events"

                current = dict(op=op)

            elif op == "equal_integer":
                current = dict(op="count_equal")
            elif op == "less_than":
                current = dict(op="count_less")
            elif op == "greater_than":
                current = dict(op="count_greater")
            else:
                raise ValueError("Unknown CLEVR operation: {}.".format(op))

        if current is None:
            assert len(block["inputs"]) == 1
            mapping[block_id] = mapping[block["inputs"][0]]
        else:
            current["inputs"] = list(map(mapping.get, block["inputs"]))

            if "_output" in block:
                current["output"] = deepcopy(block["_output"])

            nscl_program.append(current)
            mapping[block_id] = len(nscl_program) - 1

    return nscl_program


if __name__ == "__main__":
    program_gt = [
        "events",
        "objects",
        "metal",
        "filter_material",
        "sphere",
        "filter_shape",
        "unique",
        "filter_collision",
        "unique",
        "objects",
        "metal",
        "filter_material",
        "sphere",
        "filter_shape",
        "unique",
        "query_collision_partner",
        "query_shape",
    ]

    # cp = clevrer_to_clevr(program_gt)
    # print(cp)
    cp = [
        {"inputs": [], "function": "events", "value_inputs": [], "id": 0},
        {"inputs": [], "function": "events", "value_inputs": [], "id": 1},
        {"inputs": [], "function": "objects", "value_inputs": [], "id": 2},
        {"inputs": [2], "function": "filter_shape", "value_inputs": ["cube"], "id": 3},
        {"inputs": [3], "function": "unique", "value_inputs": [], "id": 4},
        {"inputs": [4, 1], "function": "filter_in", "value_inputs": [], "id": 5},
        {"inputs": [5], "function": "unique", "value_inputs": [], "id": 6},
        {"inputs": [6, 0], "function": "filter_after", "value_inputs": [], "id": 7},
        {"inputs": [], "function": "objects", "value_inputs": [], "id": 8},
        {"inputs": [8, 7], "function": "filter_collision", "value_inputs": [], "id": 9},
        {"inputs": [9], "function": "count", "value_inputs": [], "id": 10},
    ]

    cp = clevr_to_nsclseq(cp)
    print(cp)

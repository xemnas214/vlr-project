#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : definition.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/29/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Basic concepts in the CLEVR dataset.
"""

import six
import numpy as np

from jacinle.logging import get_logger
from nscl.datasets.definition import DatasetDefinitionBase
from .program_translator import clevrer_to_nsclseq

logger = get_logger(__file__)

__all__ = [
    "CLEVRERDefinition",
    "build_clevrer_dataset",
    "build_symbolic_clevrer_dataset",
    "build_concept_retrieval_clevrer_dataset",
    "build_concept_quantization_clevrer_dataset",
]


class CLEVRERDefinition(DatasetDefinitionBase):
    operation_signatures = [
        # Part 1: CLEVRER dataset.
        # Input Modules
        ("objects", [], [], "objects"),
        ("frame", [], [], "frame"),
        ("events", [], [], "events"),
        ("unseenEvents", [], [], "events"),
        ("allEvents", [], [], "events"),
        ("start", [], [], "events"),
        ("stop", [], [], "events"),
        # Object Filter Modules
        ("filter", ["attribute"], ["objects"], "objects"),
        ("filter_frame", ["objects"], ["frame"], "objects"),
        # Event Filter Modules
        ("filter_events", [], ["events", "objects"], "events"),
        ("filter_ee", ["events"], ["events"], "events"),
        ("filter_order", ["events"], ["order"], "events"),
        ("get_frame", [], ["events"], "frame"),
        ("get_counterfact", ["events"], ["objects"], "events"),
        ("get_col_partner", ["events"], ["objects"], "objects"),
        ("get_object", [], ["events"], "objects"),
        # Unique Events/ Objects
        ("unique", [], ["events"], "events"),
        ("unique", [], ["objects"], "objects"),
        # Output Modules
        ("query", ["attribute"], ["objects"], "word"),
        ("query_collision_partner", [], ["events", "objects"], "word"),
        ("count", [], ["objects"], "integer"),
        # ("count_objects", [], ["objects"], "integer"),
        # ("count_events", [], ["events"], "integer"),
        ("exist", [], ["objects"], "bool"),
        ("belong_to", ["events"], ["events"], "bool"),
        ("negate", [], ["bool"], "bool"),
    ]

    attribute_concepts = {
        "color": ["gray", "brown", "green", "red", "blue", "purple", "yellow", "cyan"],
        "material": ["metal", "rubber"],
        "shape": ["cube", "cylinder", "sphere"],
        # "size": ["small", "large"],
    }

    # events_concepts = {
    # }

    relational_concepts = {
        "count_objects": ["0", "1", "2", "3", "4", "5"],
        "count_events": ["0", "1", "2", "3", "4", "5"],
        "spatial_relation": ["left", "right", "front", "behind"],
        "order": ["first", "second", "last"],
        "moving": [],
        "stationary": [],
        "in": [],
        "out": [],
        "collision": [],
        "before": [],
        "after": [],
        "ancestor": [],
    }  ## Not required for CLEVRER

    synonyms = {
        "thing": ["thing", "object"],
        "sphere": ["sphere", "ball", "spheres", "balls"],
        "cube": ["cube", "block", "cubes", "blocks"],
        "cylinder": ["cylinder", "cylinders"],
        "large": ["large", "big"],
        "small": ["small", "tiny"],
        "metal": ["metallic", "metal", "shiny"],
        "rubber": ["rubber", "matte"],
    }  ## Keep, might need to update as per language

    word2lemma = {v: k for k, vs in synonyms.items() for v in vs}

    EBD_CONCEPT_GROUPS = "<CONCEPTS>"
    EBD_RELATIONAL_CONCEPT_GROUPS = "<REL_CONCEPTS>"
    EBD_ATTRIBUTE_GROUPS = "<ATTRIBUTES>"

    extra_embeddings = [
        EBD_CONCEPT_GROUPS,
        EBD_RELATIONAL_CONCEPT_GROUPS,
        EBD_ATTRIBUTE_GROUPS,
    ]

    @staticmethod
    def _is_object_annotation_available(scene):
        assert len(scene["objects"]) > 0
        if "mask" in scene["objects"][0]:
            return True
        return False

    def annotate_scene(self, scene):
        feed_dict = dict()

        if not self._is_object_annotation_available(scene):
            return feed_dict

        for attr_name, concepts in self.attribute_concepts.items():
            concepts2id = {v: i for i, v in enumerate(concepts)}
            values = list()
            for obj in scene["objects"]:
                assert attr_name in obj
                values.append(concepts2id[obj[attr_name]])
            values = np.array(values, dtype="int64")
            feed_dict["attribute_" + attr_name] = values
            lhs, rhs = np.meshgrid(values, values)
            feed_dict["attribute_relation_" + attr_name] = (
                (lhs == rhs).astype("float32").reshape(-1)
            )

        nr_objects = len(scene["objects"])
        for attr_name, concepts in self.relational_concepts.items():
            concept_values = []
            for concept in concepts:
                values = np.zeros((nr_objects, nr_objects), dtype="float32")
                assert concept in scene["relationships"]
                this_relation = scene["relationships"][concept]
                assert len(this_relation) == nr_objects
                for i, this_row in enumerate(this_relation):
                    for j in this_row:
                        values[i, j] = 1
                concept_values.append(values)
            concept_values = np.stack(concept_values, -1)
            feed_dict["relation_" + attr_name] = concept_values.reshape(
                -1, concept_values.shape[-1]
            )

        return feed_dict

    def annotate_question_metainfo(self, metainfo):
        if "template_filename" in metainfo:
            return dict(
                template=metainfo["template_filename"],
                template_index=metainfo["question_family_index"],
            )
        return dict()

    def annotate_question(self, metainfo):
        return dict()

    def program_to_nsclseq(self, program, question=None):
        return clevrer_to_nsclseq(program)

    def canonize_answer(self, answer, question_type):
        if answer in ("yes", "no"):
            answer = answer == "yes"
        elif isinstance(answer, six.string_types) and answer.isdigit():
            answer = int(answer)
            assert 0 <= answer <= 10
        return answer

    def update_collate_guide(self, collate_guide):
        # Scene annotations.
        for attr_name in self.attribute_concepts:
            collate_guide["attribute_" + attr_name] = "concat"
            collate_guide["attribute_relation_" + attr_name] = "concat"
        for attr_name in self.relational_concepts:
            collate_guide["relation_" + attr_name] = "concat"

        # From ExtractConceptsAndAttributes and SearchCandidatePrograms.
        for param_type in self.parameter_types:
            collate_guide["question_" + param_type + "s"] = "skip"
        collate_guide["program_parserv1_groundtruth_qstree"] = "skip"
        collate_guide["program_parserv1_candidates_qstree"] = "skip"


def build_clevrer_dataset(args, configs, image_root, scenes_json, questions_json):
    import jactorch.transforms.bbox as T

    image_transform = T.Compose(
        [
            T.NormalizeBbox(),
            T.Resize(configs.data.image_size),
            T.DenormalizeBbox(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    from nscl.datasets.datasets import NSCLDataset

    dataset = NSCLDataset(
        scenes_json,
        questions_json,
        image_root=image_root,
        image_transform=image_transform,
        vocab_json=args.data_vocab_json,
    )

    return dataset


def build_concept_retrieval_clevrer_dataset(
    args, configs, program, image_root, scenes_json
):
    import jactorch.transforms.bbox as T

    image_transform = T.Compose(
        [
            T.NormalizeBbox(),
            T.Resize(configs.data.image_size),
            T.DenormalizeBbox(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    from nscl.datasets.datasets import ConceptRetrievalDataset

    dataset = ConceptRetrievalDataset(
        program, scenes_json, image_root=image_root, image_transform=image_transform
    )
    return dataset


def build_concept_quantization_clevrer_dataset(args, configs, image_root, scenes_json):
    import jactorch.transforms.bbox as T

    image_transform = T.Compose(
        [
            T.NormalizeBbox(),
            T.Resize(configs.data.image_size),
            T.DenormalizeBbox(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    from nscl.datasets.datasets import ConceptQuantizationDataset

    dataset = ConceptQuantizationDataset(
        scenes_json, image_root=image_root, image_transform=image_transform
    )
    return dataset


def build_symbolic_clevrer_dataset(args):
    from nscl.datasets.datasets import NSCLDataset

    dataset = NSCLDataset(
        args.data_scenes_json,
        args.data_questions_json,
        image_root=None,
        image_transform=None,
        vocab_json=args.data_vocab_json,
    )

    return dataset

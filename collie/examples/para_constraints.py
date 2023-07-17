"""Paragraph generation constraints"""
import random
from collie.constraints import *
from ..extractor_utils import ConstraintExtractor, raise_exception


# sentence count constraint
PARA_CONSTRAINTS = {
    "c08": ConstraintExtractor(
        init_range = {
            "input_level": [InputLevel("sentence")],
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(Position(0))],
            "relation": [Relation("==")],
            "reduction": [Reduction("all")]
        },
        post_extract=lambda x: x[0] if len(set(x)) == 1 and len(x) > 1 else raise_exception()
    ),
    "c09": [
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("sentence")],
                "transformation": [Count()],
                "relation": [Relation("==")]
            },
            target_range = list(range(3, 10))
        ),
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [ForEach(...)],
                "relation": [Relation("not in")],
            },
            target_range=["the", "be", "there"]
        ),
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [ForEach(...)],
                "relation": [Relation("not in")],
            },
            target_range=["of", "to", "this"] 
        ),
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [ForEach(...)],
                "relation": [Relation("not in")],
            },
            target_range=["and", "in", "is"] 
        )
    ],
    "c10": [
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("sentence")],
                "transformation": [Count()],
                "relation": [Relation("==")]
            },
            target_range = list(range(3, 10))
        ),
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("sentence")],
                "target_level": [TargetLevel("word")],
                "transformation": [ForEach(Count())],
                "relation": [Relation(">=")],
                "reduction": [Reduction("all")]
            },
            post_extract=lambda x: min(x) if min(x) >= 10 else raise_exception() 
        ),
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("sentence")],
                "target_level": [TargetLevel("word")],
                "transformation": [ForEach(Count())],
                "relation": [Relation("<=")],
                "reduction": [Reduction("all")]
            },
            post_extract=lambda x: max(x) if max(x) <= 20 else raise_exception() 
        ),
    ],
    "c11": [
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("sentence")],
                "transformation": [Count()],
                "relation": [Relation(">=")]
            },
            target_range = [3]
        ),
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("sentence")],
                "target_level": [TargetLevel("word")],
                "transformation": [ForEach(Count())],
                "relation": [Relation(">=")],
                "reduction": [Reduction("all")]
            },
            post_extract=lambda x: min(x) if min(x) > 15 else raise_exception() 
        ),
    ],
    "c12": [
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("sentence")],
                "transformation": [Count()],
                "relation": [Relation("==")]
            },
            target_range = list(range(3, 6))
        ),
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("sentence")],
                "target_level": [TargetLevel("word")],
                "transformation": [ForEach(Position(-1))],
                "relation": [Relation("==")],
                "reduction": [Reduction("all")]
            },
            target_range = None
        ),
    ],
}

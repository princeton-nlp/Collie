"""Sentence generation constraints"""
import random
from collie.constraints import *
from ..extractor_utils import ConstraintExtractor, raise_exception


# sentence count constraint
SENT_CONSTRAINTS = {
    "c04": ConstraintExtractor(
        init_range = {
            # "input_level": [InputLevel("sentence")],
            "target_level": [TargetLevel("character")],
            "transformation": [Count()],
            "relation": [Relation("==")]
        },
        target_range = list(range(80,120))
    ),
    "c05": [
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [Count()],
                "relation": [Relation("==")]
            },
            target_range = list(range(10,20))
        ),
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [Position([3, 7, 10])],
                "relation": [Relation("==")],
            },
            target_range = None
        )
    ],
    "c06a": [
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [Count()],
                "relation": [Relation(">=")]
            },
            post_extract=lambda x: x if x > 7 else raise_exception()
            # post_extract = lambda x: breakpoint()
        ),
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("word")],
                "target_level": [TargetLevel("character")],
                "transformation": [ForEach(Count())],
                "relation": [Relation("<=")],
                "reduction": [Reduction("all")]
            },
            # this will pull out all examples less than 7 and reject all others
            post_extract=lambda x: max(x) if max(x) < 8 else raise_exception() 
            # post_extract=lambda x: breakpoint()
        )
    ],
    "c07": ConstraintExtractor(
        init_range = {
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(...)],
            "relation": [Relation("in")],
        },
        post_extract=lambda x: random.sample(x, 3) if len(x) >= 3 else raise_exception()
    )
}

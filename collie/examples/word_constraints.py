"""Word generation constraints"""
import random
from collie.constraints import *
from ..extractor_utils import ConstraintExtractor, raise_exception

# sentence count constraint
WORD_CONSTRAINTS = {
    "c01": ConstraintExtractor(
        init_range = {
            "target_level": [TargetLevel("character")],
            "transformation": [Count()],
            "relation": [Relation(">=")],
        },
        post_extract=lambda x: x if x >= 10 else raise_exception() 
    ),
    "c02": [
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("character")],
                "transformation": [Count()],
                "relation": [Relation("==")]
            },
            post_extract=lambda x: x if x >= 12 else raise_exception() 
        ),
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("character")],
                "transformation": [Position([3, 7, 10])],
                "relation": [Relation("==")],
            },
            target_range = None
        )
    ],
    "c03": [
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("character")],
                "transformation": [Count()],
                "relation": [Relation("==")]
            },
            post_extract=lambda x: x if x >= 7 else raise_exception() 
        ),
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("character")],
                "transformation": [Position(-1)],
                "relation": [Relation("==")],
            },
            target_range = ["z", "q", "x", "c", "b", "i", "v"]
        )
    ],
}

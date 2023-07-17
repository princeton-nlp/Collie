"""Paragraph generation constraints"""
import random
from collie.constraints import *
from ..extractor_utils import ConstraintExtractor, raise_exception, no_sents_filter


# sentence count constraint
PASSAGE_CONSTRAINTS = {
    # "c13": [
    #     ConstraintExtractor(
    #         init_range = {
    #             "target_level": [TargetLevel("paragraph")],
    #             "transformation": [Count()],
    #             "relation": [Relation("==")]
    #         },
    #         post_extract=lambda x: x if x in range(3,5) else raise_exception() 
    #     ),
    #     ConstraintExtractor(
    #         init_range = {
    #             "input_level": [InputLevel("paragraph")],
    #             "target_level": [TargetLevel("word")],
    #             "transformation": [ForEach(Count())],
    #             "relation": [Relation("==")],
    #         },
    #     ),
    # ],
    "c14": [
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("paragraph")],
                "target_level": [TargetLevel("sentence")],
                "transformation": [ForEach(Position(-1))],
                "relation": [Relation("==")],
                "reduction": [Reduction("all")]
            },
            # make sure there are between 2 and 5 paragraphs, and each sentence target is not too long
            post_extract=lambda x: x if len(x) in range(2, 5) else raise_exception() #and all([len(y.split(" ")) < 20 for y in x]) else raise_exception()
        ),
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("paragraph")],
                "target_level": [TargetLevel("sentence")],
                "transformation": [ForEach(Count())],
                "relation": [Relation(">=")],
                "reduction": [Reduction("all")]
            },
            target_range=[2]
        ),
    ]
}

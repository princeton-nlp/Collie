"""Loader and extractor for CC-News"""
from pathlib import Path
import random
import copy
from typing import List, Callable
from datasets import load_dataset, Dataset

from .extractor_utils import TextLoader


class EnglishLoader(TextLoader):
    """class for iterating over Gutenberg, dammit files.""" 
    def __init__(self, cache_dir:str=None, randomize:bool=False, **kwargs):
        self.randomize = randomize
        self.wordlist = Path(cache_dir).joinpath("english3.txt").read_text()

    def __iter__(self):
        def word_iter():
            yield self.wordlist, dict() 
        self._word_iter = word_iter()
        return self
    
    def __next__(self):
        txt, meta = next(self._word_iter)
        return txt, meta
    
    def __len__(self):
        return len(self.wordlist)


if __name__ == "__main__":
    from rich import print
    from .constraints import (
        Constraint,
        TargetLevel,
        Count,
        Relation,
        Position,
        Reduction
    )
    from .extractor_utils import TextChunker, ConstraintExtractor
    from .extract_constraints import FullExtractor

    textloader = EnglishLoader(
        cache_dir="./data",
        randomize=True
    )

    chunker = TextChunker(
        paragraph_delim="\n",
        randomize=True,
    )

    extractor = FullExtractor(
        chunker = chunker,
        loader = textloader,
    )
    # for txt, m in textloader:
    #     for seq in chunker(txt):
    #         print(seq + "\n")
    #         input()


    # First constraint
    constr_extractor = ConstraintExtractor(
        init_range = {
            "target_level": [TargetLevel("character")],
            "transformation": [Count()],
            "relation": [Relation("==")]
        },
        target_range = list(range(5,15))
    )

    extractor.extract(constr_extractor, max_seq_per_document=100)
    extractor.print_examples(num=3)

    # Second constraint
    constr_extractor = ConstraintExtractor(
        init_range = {
            "target_level": [TargetLevel("character")],
            "transformation": [Position([0, 1, 3]), Position([3, 5])],
            "relation": [Relation("==")],
            "reduction": [Reduction("all")]
        },
        target_range = None
    )

    extractor.extract(constr_extractor, max_seq_per_document=100)
    extractor.print_examples(num=3)
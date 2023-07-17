"""Loader and extractor for CC-News"""
import random
import copy
from typing import List, Callable
from datasets import load_dataset, Dataset

import nltk
from .extractor_utils import (
    TextLoader,
    chain_filters,
    url_filter,
    copyright_filter,
    caption_filter,
    no_sents_filter
)


def get_ccnews_filter() -> Callable[[str], bool]:
    return chain_filters(
        copyright_filter,
        url_filter,
        caption_filter,
        no_sents_filter
    )


class CCNewsLoader(TextLoader):
    """class for iterating over Gutenberg, dammit files.""" 
    def __init__(self, cache_dir:str=None, randomize:bool=False, **kwargs):
        self.randomize = randomize
        self.dataset:Dataset = load_dataset(
            "cc_news",
            split="train",
            cache_dir=cache_dir,
            **kwargs
        )
        self.indices:List = None

    def __iter__(self):
        self.indices = list(range(len(self.dataset)))
        if self.randomize:
            random.shuffle(self.indices)
        def metadata_iter():
            for idx in self.indices:
                datadict = copy.deepcopy(self.dataset[idx])
                txt = datadict.pop("text")
                datadict["index"] = idx
                yield txt, datadict 
        self._metadata_iter = metadata_iter()
        return self
    
    def __next__(self):
        txt, meta = next(self._metadata_iter)
        return txt, meta
    
    def __len__(self):
        return len(self.dataset)


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

    textloader = CCNewsLoader(
        cache_dir="./data",
        randomize=True
    )

    chunker = TextChunker(
        paragraph_delim="\n",
        randomize=True,
        filter=get_ccnews_filter(),
        chunk_by_passage=True,
    )

    extractor = FullExtractor(
        chunker = chunker,
        loader = textloader,
        metadata_fields=("index", "title")
    )
    for txt, m in textloader:
        for seq in chunker(txt):
            print(seq + "\n")
            input()


    # First constraint
    constr_extractor = ConstraintExtractor(
        init_range = {
            "target_level": [TargetLevel("word")],
            "transformation": [Count()],
            "relation": [Relation("==")]
        },
        target_range = list(range(5,15))
    )

    extractor.extract(constr_extractor, max_documents=10)
    extractor.print_examples(num=3)

    # Second constraint
    constr_extractor = ConstraintExtractor(
        init_range = {
            "target_level": [TargetLevel("word")],
            "transformation": [Position([0, 1, 3]), Position([3, 5, 9])],
            "relation": [Relation("==")],
            "reduction": [Reduction("all")]
        },
        target_range = None
    )

    extractor.extract(constr_extractor, max_documents=10)
    extractor.print_examples(num=3)
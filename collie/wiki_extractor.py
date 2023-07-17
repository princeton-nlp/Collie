"""Loader and extractor for Wikipedia"""
import random
import copy
from typing import List
from datasets import load_dataset, Dataset

from .extractor_utils import (
    TextLoader,
    no_sents_filter,
    url_filter,
    caption_filter,
    chain_filters
)


def get_wiki_filter():
    likely_table = lambda x: "|" in x
    return chain_filters(
        likely_table,
        url_filter,
        caption_filter,
        no_sents_filter
    )


def get_wiki_postprocessor():
    def _wiki_postprocessor(text:str):
        if "\n" not in text:
            return text
        return " ".join(text.split("\n")[1:]) # get rid of section titles
    return _wiki_postprocessor


class WikiLoader(TextLoader):
    """class for iterating over Wikipedia Articles.""" 
    def __init__(self, cache_dir:str=None, randomize:bool=False, split="train", **kwargs):
        self.randomize = randomize
        self.dataset:Dataset = load_dataset(
            "wikipedia",
            "20220301.en",
            cache_dir=cache_dir,
            split=split,
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
    # dataset = load_dataset("wikipedia", "20220301.en", cache_dir="./data")
    # breakpoint()
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

    textloader = WikiLoader(
        cache_dir="./data",
        randomize=True
    )

    chunker = TextChunker(
        paragraph_delim="\n\n",
        randomize=True,
        filter=get_wiki_filter(),
        postprocessor=get_wiki_postprocessor(),
        chunk_by_passage=True,
    )

    extractor = FullExtractor(
        chunker = chunker,
        loader = textloader,
        metadata_fields=("index", "title")
    )

    from pathlib import Path
    for txt, m in textloader:
        for seq in chunker(txt):
            print(seq + "\n")
            Path("dump.txt").write_text(seq)
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
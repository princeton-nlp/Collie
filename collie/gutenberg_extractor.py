"""Scripts to load and extract constraints from Gutenberg, dammit."""

from typing import Callable, Iterable
from pathlib import Path
import json
import random
from tqdm.autonotebook import tqdm

from .extractor_utils import (
    TextLoader,
    markdown_remover,
    consecutive_whitespace_remover,
    replace_singlenewline_with_space,
    remove_brackets,
    all_caps_filter,
    no_sents_filter,
    chain_processors,
    chain_filters
)


def gutenberg_preprocessor(text:str):
    raise Exception("this does something weird with the chunking. ignore for now.")
    text = markdown_remover(text)
    return consecutive_whitespace_remover(text)

def get_gutenberg_postprocessor() -> Callable[[str], str]:
    return chain_processors(
        markdown_remover,
        remove_brackets,
        replace_singlenewline_with_space,
        consecutive_whitespace_remover
    )

def get_gutenberg_filter() -> Callable[[str], bool]:
    return chain_filters(
        all_caps_filter, # all caps indicate section title
        no_sents_filter
    )

class GutenbergLoader(TextLoader):
    """class for iterating over Gutenberg, dammit files.""" 
    def __init__(self, filepath:str, randomize:bool=False):
        self.filepath = Path(filepath)
        self.datadir = Path(filepath).parent
        self.randomize = randomize
        with self.filepath.open(mode="r") as f:
            self.metadata = json.load(f)
        self._metadata_iter = None

    def __iter__(self):
        if self.randomize:
            random.shuffle(self.metadata)
        def metadata_iter():
            for m in self.metadata:
                yield m["gd-path"], m
        self._metadata_iter = metadata_iter()
        return self
    
    def __next__(self):
        lang = None 
        while lang != ["English"]:
            path, m = next(self._metadata_iter)
            lang = m["Language"]
        txt = self.datadir.joinpath(path).read_text()
        return txt, m
    
    def __len__(self):
        return len(self.metadata)


if __name__ == "__main__":
    from rich import print
    from collie.constraints import (
        Constraint,
        TargetLevel,
        Count,
        Relation,
        Position,
        Reduction
    )
    from .extractor_utils import TextChunker, ConstraintExtractor
    from .extract_constraints import FullExtractor

    textloader = GutenbergLoader(
        "data/gutenberg-dammit-files/gutenberg-metadata.json",
        randomize=True
    )

    chunker = TextChunker(
        paragraph_delim="\n\n",
        postprocessor=get_gutenberg_postprocessor(),
        filter=get_gutenberg_filter(),
        randomize=True,
        chunk_by_passage=True,
    )

    extractor = FullExtractor(
        chunker = chunker,
        loader = textloader,
        metadata_fields=("Title", "Author","gd-path")
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
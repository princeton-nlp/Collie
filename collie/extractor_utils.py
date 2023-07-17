"""Code for preprocessing text"""
import copy
from typing import Type, Iterable, Callable, Dict, Any, Tuple
import random
from collections import OrderedDict, defaultdict
import re
import functools
import itertools
import nltk
from .constraints import *


# useful for post_extract lambdas in ConstraintExtractor.
def raise_exception():
    raise Exception()


# Functions for TextChunker postprocessor/preprocessor arguments
def markdown_remover(text:str) -> str:
    return re.sub(r'(\*\*|__|\*|_|\~\~)(.*?)\1', r'\2', text)


def consecutive_whitespace_remover(text:str) -> str:
    return re.sub(r'\s{2,}', ' ', text)


def replace_singlenewline_with_space(text:str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_references(text:str) -> str:
    # remove things like [23]
    return re.sub(r'\[\d+\]', '', text)


def remove_brackets(text:str) -> str:
    # remove anything between two square brackets.
    return re.sub(r'\[[^\]]*\]', '', text)


def chain_processors(*funcs:Iterable[Callable[[str], str]]) -> Callable[[str], str]:
    compose = lambda f, g: lambda text: f(g(text)) # compose two functions
    return functools.reduce(compose, funcs) # apply compose to all funcs


# Functions for TextChunker filter argument
def all_caps_filter(text:str) -> bool:
    # return True if text is all caps
    return text.upper() == text


def url_filter(text:str) -> bool:
    # return True if there is a url in the text
    pattern = re.compile(r"(http(s)?://)?(www\.)?[a-zA-Z0-9\-]+\.[a-zA-Z]{2,6}(\.[a-zA-Z]{2,6})?(/[a-zA-Z0-9\-]*)*(\?[a-zA-Z0-9\-=&]*)?")
    return bool(re.findall(pattern, text))


def no_sents_filter(text:str) -> bool:
    # returns true if no sentence is detected in the text
    for s in nltk.sent_tokenize(text):
        if "." in s and len(s.split()) > 2: # could be a sentence
            return False
    return True


def copyright_filter(text:str) -> bool:
    # possibly a copyright watermark at bottom of the page
    return "©" in text or text.split()[0].lower() == "copyright"


def caption_filter(text:str) -> bool:
    # filter potential image or text captions
    return len(text.split(":")[0].split()) <= 5 


def chain_filters(*funcs:Iterable[Callable[[str], bool]]) -> Callable[[str], bool]:
    # chain multiple filters together to make new function
    return lambda text: any([f(text) for f in funcs])


# Functions for ConstraintExtractor init_modifier argument
def add_units_to_count(init_range, text):
    # automatically fill in transformation Count(...) based on text
    assert len(init_range["target_level"]) == 1, "Multiple target levels, is not defined for Count"
    level = init_range["target_level"][0]
    units = level(text) # tokenize based on level
    init_range["transformation"] = [Count(u) for u in units]
    return init_range 


class TextLoader(Iterable):
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[str, Any]:
        # should return a piece of text and the metadata associated with the text
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    

class TextChunker(Iterable):
    """Iterable class for chunking text returned by a TextLoader"""
    def __init__(
        self,
        paragraph_delim:str="\n",
        preprocessor:Callable=lambda x: x,
        postprocessor:Callable=lambda x: x,
        filter:Callable=lambda x: False,
        randomize:bool=False,
        chunk_by_sentence:bool=False,
        chunk_by_passage:bool=False,
    ):
        self.paragraph_delim = paragraph_delim
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.filter = filter
        self.randomize = randomize
        self.text = None
        self._seq_iter = None
        self._len = None
        assert not (chunk_by_passage and chunk_by_sentence)
        self.chunk_by_sentence = chunk_by_sentence
        self.chunk_by_passage = chunk_by_passage
        self.rejected, self.total = 0.0, 0.0
        if self.chunk_by_sentence:
            nltk.download("punkt") 

    def __call__(self, text):
        self.text = self.preprocessor(text)
        return self
    
    @property
    def percent_rejected(self):
        return self.rejected / self.total
    
    def reset_stats(self):
        self.rejected, self.total = 0.0, 0.0
    
    def _get_passage_chunks(self, paragraphs:Iterable[str]) -> Iterable[str]:
        buffer = [] # list of paragraphs, that may be added to passages
        passages = [] # list of passages
        for p in paragraphs:
            if self.filter(p): # did not make the cut
                self.total += 1
                if len(buffer) > 2: # see if we should add buffer to passages
                    passages.append(Level.join_paragraphs(buffer))
                else:
                    self.rejected += 1
                buffer = []
            else: # paragraph not filtered so add it to the buffer
                buffer.append(self.postprocessor(p))
        if len(buffer) > 2: # catch last group of paragraphs
            passages.append(Level.join_paragraphs(buffer))
        return passages
    
    def __iter__(self):
        paragraphs = re.split(self.paragraph_delim, self.text)
        if self.chunk_by_sentence:
            sequences = []
            for chunk in paragraphs:
                sequences.extend(nltk.sent_tokenize(chunk))
        elif self.chunk_by_passage:
            sequences = self._get_passage_chunks(paragraphs)
        else:
            sequences = paragraphs
        if self.randomize:
            random.shuffle(sequences)
        self._len = len(sequences)
        def seq_iter():
            for s in sequences:
                yield s
        self._seq_iter = seq_iter()
        return self

    def __next__(self):
        if self.chunk_by_passage: # in this case filter and post already run.
            return next(self._seq_iter) 
        seq = ""
        total = 0
        while not seq or self.filter(seq):
            seq = next(self._seq_iter)
            total += 1
        self.total += total 
        self.rejected += total - 1
        return self.postprocessor(seq)

    def __len__(self):
        return self._len 


# get stats 
def get_stats(loader:TextLoader, chunker:TextChunker, stats:Dict[str, Callable[[str], Any]], max_passages=None):
    aggregate = defaultdict(list)
    loader = itertools.islice(loader, 0, max_passages) if max_passages is not None else loader
    for txt, m in loader:
        for seq in chunker(txt):
            for k, stat in stats.items():
                aggregate[k].append(stat(seq))
    aggregate.update({"percent_rejected": chunker.percent_rejected})
    return aggregate


class ConstraintExtractor(Iterable):
    """Extracts constraints from text returned by TextChunker. Iterates over set of instantiation time parameters and inference-time targets to check truth-value for a given constraint class and text.
    """
    def __init__(
        self,
        init_range:Dict[str, Iterable],
        target_range:Iterable = None,
        ConstraintCls:Type[Constraint] = Constraint,
        post_extract:Callable = lambda x: x,
        init_modifier:Callable = None
    ):
        self.ConstraintCls = ConstraintCls
        self._init_range = OrderedDict(**init_range) # store the original 
        self.init_range = None # can change on each __call__ 
        self.target_range = target_range
        self._combined_iter = None
        self.post_extract = post_extract
        self.init_modifier = init_modifier

    def __call__(self, text):
        self.text = text
        if self.init_modifier is not None:
            self.init_range = self.init_modifier(copy.deepcopy(self._init_range), text)
        else:
            self.init_range = self._init_range
        return self

    def __iter__(self):
        val_iter = itertools.product(*self.init_range.values())
        def combined_iter():
            for vals in val_iter:
                if self.target_range is None:
                    yield {k:v for k, v in zip(self.init_range.keys(), vals)}, None
                else:
                    for target in self.target_range:
                        yield {k:v for k, v in zip(self.init_range.keys(), vals)}, target
        self._combined_iter = combined_iter()
        return self
    
    def __next__(self):
        kwargs, target = next(self._combined_iter)
        constraint = self.ConstraintCls(**kwargs)
        if self.target_range is None:
            try:
                self.post_extract(constraint.extract(self.text))
            except: # if failed to extract example from text
                return False, (constraint, None)
            return True, (constraint, self.post_extract(constraint.extract(self.text)))
        return constraint(self.text, target), (constraint, target)
       
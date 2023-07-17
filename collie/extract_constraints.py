"""Extract constraints from data source to `data/` or `sample_data/`."""
import json
from typing import Dict, List, Any, Iterable, Union
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import dill 
import itertools
from tqdm.autonotebook import tqdm
from rich import print
import random

from .constraints import *
from .extractor_utils import ConstraintExtractor, TextChunker, TextLoader
from .constraint_renderer import ConstraintRenderer


@dataclass
class Support:
    """ A single supporting example for a constraint
    """
    constraint: Constraint
    target: Any 
    example: str
    metadata: dict


class FullExtractor:
    """ Full end-to-end constraint extraction, including prompt rendering and writing/formatting results.
    """
    def __init__(self,
        chunker:TextChunker,
        loader:TextLoader,
        metadata_fields:Iterable=None
    ):
        self.chunker = chunker
        self.loader = loader 
        self.metadata_fields = metadata_fields
        self.results:Dict[str,List[List[Support]]] = None # [example][constraint_idx][support_idx]

    def _all_sat(self, extractor, seq):
        # check that all extractors have at least one satisfying configuration
        if len(extractor) == 1:
            return True
        for ext in extractor:
            if not any([x[0] for x in ext(seq)]):
                return False
        return True
 
    def extract(
        self,
        constraints:Union[ConstraintExtractor, List[ConstraintExtractor]],
        max_documents:int=None,
        max_seq_per_document:int=None,
        conjunction:bool=True, # if set to True, every Support requires all constraints to have at least one satisfied target.
    ):
        if not (isinstance(constraints, list) or isinstance(constraints, tuple)):
            constraints = [constraints]

        self.results = defaultdict(lambda: [[] for _ in range(len(constraints))])
        passage_iter = itertools.islice(self.loader, 0, max_documents) if max_documents is not None else self.loader
        for passage, metadata in tqdm(passage_iter, total=max_documents, leave=False):
            if self.metadata_fields is not None:
                metadata = {k: metadata[k] for k in self.metadata_fields if k in metadata}
            seq_iter = itertools.islice(
                self.chunker(passage), 0, max_seq_per_document
            ) if max_seq_per_document is not None else self.chunker(passage)
            for seq in seq_iter:
                # first check that all constraints has a target that works with this seq
                if conjunction and not self._all_sat(constraints, seq):
                    continue

                for i, ext in enumerate(constraints):
                    for sat, (constraint, target) in ext(seq):
                        if not sat: continue
                        self.results[seq][i].append(
                            Support(
                                constraint=constraint,
                                target=target,
                                example=seq,
                                metadata=metadata
                            )
                        )

    def save(self, filepath:str):
        with Path(filepath).open(mode="wb") as f:
            dill.dump(self.results, f)

    @staticmethod
    def load(self, filepath:str):
        with Path(filepath).open(mode="rb") as f:
            return dill.load(f)
        
    def print_examples(self, num:int=1, conjunction:bool=False):
        for ex in self.get_constraints(total_examples=num, conjunction=conjunction):
            print(ex)
            print()

    def _get_prompt(self, constraint, target):
        renderer = ConstraintRenderer(constraint, target)
        return renderer.prompt
    
    def inspect_results(self, file:str="extractor_results.txt"):
        result = ""
        for ex in self.results.keys():
            result += f"\n---------------\n{ex}\n----------------\n"
            for i, constrs in enumerate(self.results[ex]):
                result += f"\tConstraint: {i}\n\n"
                for sup in constrs:
                    result += f"\t\tSupport: {sup.constraint}\n"
                    result += f"\t\tTarget: {sup.target}\n\n"
        Path(file).write_text(result)

    def get_constraints(self, total_examples:int, conjunction:bool=False):
        # get list of prompts, examples, and metadata.
        # need this because naive combinatorial sampling is too slow
        results = []
        sampled_idx, total, attempts = set(), 0, 0
        while total < total_examples:
            attempts += 1
            if attempts > 2 * total_examples:
                break
            # choose random example
            try: 
                ex = random.choice(list(self.results.keys()))
            except:
                return results # self.results is probably empty.
            # choose supports for this example. If conjunction, choose one support for each constraint
            if conjunction:
                # for that example, choose supports from each constraint in self.results[ex]
                idx = tuple([random.choice(range(len(x))) for x in self.results[ex]])
                if (ex, idx) in sampled_idx:
                    continue
                sampled_idx.add((ex, idx))
                supports = [x[i] for x, i in zip(self.results[ex], idx)]

                # each support should correspond to the same example
                assert len(set([s.example for s in supports])) == 1

            else: 
                # choose a random single constraint
                constr_idx = random.choice(range(len(self.results[ex])))
                # choose a support for that constraint
                if len(self.results[ex][constr_idx]) < 1:
                    continue
                support_idx = random.choice(range(len(self.results[ex][constr_idx])))
                if (ex, constr_idx, support_idx) in sampled_idx:
                    continue
                sampled_idx.add((ex, constr_idx, support_idx)) 
                supports = [self.results[ex][constr_idx][support_idx]]

            # render the prompts
            targets = [s.target for s in supports]
            constraints = [s.constraint for s in supports]
            constraint, target = (All(*constraints), targets) if len(constraints) > 1 else (constraints[0], targets[0])
            # prompt = self._get_prompt(constraint, target)
            metadata = supports[0].metadata
            total += 1 

            results.append({
                # "prompt": prompt, 
                "example": ex,
                "metadata": metadata,
                "targets": targets,
                "constraint": constraint
            })
        return results           

    def write_results(self, file:str, conjunction:bool=False, total_examples:int=100, extra_info:dict=None):
        raise NotImplementedError
        results = self.get_constraints(total_examples=total_examples, conjunction=conjunction)
        Path(file).write_text(results) 

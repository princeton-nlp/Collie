"""Full extraction on examples"""
import itertools
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union
import dill
from ..extract_constraints import FullExtractor
from ..extractor_utils import ConstraintExtractor
from .sent_constraints import SENT_CONSTRAINTS
from .para_constraints import PARA_CONSTRAINTS
from .word_constraints import WORD_CONSTRAINTS
from .passage_constraints import PASSAGE_CONSTRAINTS
from .sources import SOURCE_WORD_EXTRACTORS, SOURCE_SENT_EXTRACTORS, SOURCE_PARA_EXTRACTORS, SOURCE_PASSAGE_EXTRACTORS


def extract_all(
    outdir:str, 
    extractors: Dict[str, FullExtractor],
    constraints: Dict[str, Union[List[ConstraintExtractor], ConstraintExtractor]],
    max_passage:int=300,
    max_seq_per_passage:int=100,
    ex_per_constraint:int=100,
    suffix:str = "",
    conj=True
):
    results = defaultdict(dict)
    for (source, extractor), (constr_name, constraint) in itertools.product(
        extractors.items(), constraints.items()
    ):
        extractor.extract(constraint, conjunction=conj, max_documents=max_passage, max_seq_per_document=max_seq_per_passage)
        results[source][constr_name] = extractor.get_constraints(total_examples=ex_per_constraint, conjunction=conj)
        # extractor.inspect_results(f"temp/{source}{suffix}_dump.txt")
    
    for source, r in results.items():
        with Path(outdir).joinpath(f"{source}{suffix}.dill").open(mode="wb") as f:
            dill.dump(r, f)


if __name__ == "__main__":
    extract_all(
        "sample_data",
        extractors=SOURCE_SENT_EXTRACTORS,
        constraints=SENT_CONSTRAINTS,
        suffix="_sent",
    )

    extract_all(
        "sample_data",
        extractors=SOURCE_PARA_EXTRACTORS,
        constraints=PARA_CONSTRAINTS,
        suffix="_para",
    )

    extract_all(
        "sample_data",
        extractors=SOURCE_WORD_EXTRACTORS,
        constraints=WORD_CONSTRAINTS,
        suffix="_word",
        max_seq_per_passage=None,
    )

    extract_all(
        "sample_data",
        extractors=SOURCE_PASSAGE_EXTRACTORS,
        constraints=PASSAGE_CONSTRAINTS,
        suffix="_passage",
        max_seq_per_passage=None,
    )
"""Full extraction on examples"""
from pathlib import Path
from typing import Any, Dict, List
import dill


def validate_example(file:str):
    # results is a dictionary. First key is the ID of the constraint (e.g. c10). This gives you a list of 
    # dictionaries, each containing information about the prompt, example, target, and constraint object.
    with Path(file).open(mode="rb") as f:
        results:Dict[str, List[Dict[str, Any]]] = dill.load(f)

    # here is an example of how you would validate a model generation
    for constraint_id, examples in results.items():
        for x in examples: 
            assert isinstance(x["prompt"], str) # this is the prompt
            assert isinstance(x["example"], str) # this is the example

            # let's validate the example, clearly it should satisfy the constraint. 
            # you can replace ex with the model generation to see if it fit the constraint.
            ex = x["example"]
            assert x["constraint"](ex, x["targets"]) # x["constraint"] is the actual Constraint or All object.
    print(f"All validations passed!") 

# Extraction Usage
This document details how to use the extraction features of `collie` to get constraint targets from data sources. The features are still WIP and are subject to change.

## Preparing Data Sources
First we need to load our data sources. This is done using the following utilities:

### TextLoader
This should be an iterable that returns documents from the data source and should inherit from `collie.extractor_utils.TextLoader`. For instance, see `collie.wiki_extractor.WikiLoader`.

### TextChunker
Constraints often operate at a level lower than an entire document, and the chunker splits the document into paragarphs, sentences, or passages (multiple consecutive paragraphs). The chunker is itself also an iterable, and should directly use, or inherit from `collie.extractor_utils.TextChunker`. Our default chunker has the following arguments:
- `paragraph_delim`: This tells the chunker what delimiter is used to separate paragraphs within the document. For example, in Wikipedia this is a double newline. 
- `preprocessor`: This is an optional callable that ingests the entire document and runs preprocessing on it before any chunking is done. 
- `postprocessor`: Optional callable that modifies the chunks returned by the chunker. 
- `filter`: Optional callable that takes as input one of the chunked examples, and returns `True` if this example should be rejected. Returns `False` otherwise. Note the usage of the filter is special for passages, see `chunk_by_passage`.
- `randomize`: Randomize the order of the returned sequences.
- `chunk_by_sentence`: If set to `True`, run `nltk.sent_tokenize` on each paragraph, and return sentences.
- `chunk_by_passage`: If set to `True`, extract passages from the document. This is done by going paragraph by paragraph through the document, running the `filter` on each, and returning as many consecutive paragraphs that pass filtering as possible. Passages of a single paragraph are omitted.

Here is an example of how the loader and chunker works in conjunction:
```python
# example illustrating how the loader and chunker works
from collie.wiki_extractor import WikiLoader
from collie.extractor_utils import TextChunker

postprocess = lambda x: x + " :)" # add a smiley face to the end of every chunk
f = lambda x: ":(" in x # reject any chunk with a frowny face in it

loader = WikiLoader() 
chunker = TextChunker( 
    paragraph_delim="\n\n",
    chunk_by_sentence=True,
    postprocessor=postprocess,
    filter=f
)

for document, metadata in loader:
    for sentence in chunker(document):
        print(sentence)
```
The difficulty with adding new sources typically comes from defining high-recall filters and good post-processors that can remove artifacts like markdown formatting. There are some off-the-shelf filters and processors defined in `collie.extractor_utils`. You can chain multiple filters and processors using `chain_processors` and `chain_filters`. 

For example loaders and chunkers see `collie/examples/sources.py`.

## Constraint Extraction
Now that we have sequences, we can use these to extract the targets for our constraints. This is done using the following tools:

### ConstraintExtractor
This class is an iterable that iterates over different configurations of the constraints, including initialization parameters, and target values, and indicates which configurations are satisfied by a source input. The class has the following main initialization options:
- `init_range`: This is a dict of iterables. The extractor will generate a full combinatorial grid from all the iterables, each of which instantiates a unique `Constraint` object.
- `target_range`: If specified, this should be an iterable that uniquely defines the range of targets to sweep over. If this is `None`, then the output of `Constraint.extract()` is used as the target value. Using the `extract()` method should be used especially when the target range to sweep is excessively large.
- `post_extract`: A callable that modifies the returned target. Should only be used if `target_range` is `None`. It can optionally reject the example by raising an exception.
- The iterable returns a tuple that consists of `(is_satisfied, (constraint, target))` where `is_satisfied` indicates whether the constraint is satisfied, `constraint` is the actual `Constraint` object, and `target` is the extracted target value.

Here are some example extractors.
```python
from collie.extractor_utils import ConstraintExtractor
from collie.constraints import Constraint

init_range = {
    "target_level": [TargetLevel("word")],
    "transformation": [Position([3, 7, 10]), Position([1, 2, 9])],
    "relation": [Relation("==")],
},

# this extractor will initialize two Constraint objects, one looking at positions (3, 7, 10) and the other at (1, 2, 9)
constr_1 = ConstraintExtractor(
    init_range = init_range,
    target_range = [["the", "duh", "bruh"]] # only check if it satisfies this one target. 
)

# same as above, but use extract and replace all target instances of "the" with "duh" and reject the example if "bruh" is a target.
def post(target):
    if "bruh" in target:
        raise Exception()
    return [x if x != "the" else "duh" for x in target]

constr_2 = ConstraintExtractor(
    init_range = init_range,
    target_range = None, # just extract the words, no need to sweep anything
    post_extract = post
)
```
Here is a demo of how the extractor works with the loader and chunker defined above:
```python
for document, metadata in loader:
    for sentence in chunker(document):
        for is_satisfied, (constraint, target) in constr_1(sentence):
            assert isinstance(constraint, Constraint) # true
            if is_satisfied:
                print(target)
```

Constraints we have defined so far can be found in `collie/examples/*_constraints.py`.

### FullExtractor
While it is possible to use the loop above and write your own extraction logic using it, we have defined utilities that may make life easier.

The class, `collie.extract_constraints.FullExtractor` combines the loader and the chunker into a single pipeline, and offers additional functions during constraint extraction, like limiting the maximum number of chunks each passage can contribute to the extraction process. The init arguments for this class are as follows:
- `chunker`: the `TextChunker` to use.
- `loader`: the `TextLoader` to use.
- `metadata_fields: an iterable of strings, specifying the metadata fields returned by the loader to store. 

Using the same loader and chunker from above, we can build a `FullExtractor` pipeline as follows:
```python
extractor = FullExtractor(
    chunker = chunker,
    loader = loader,
    metadata_fields=("index", "title")
)
```

The `FullExtractor` object has a few core methods:
- `extract()`: Passing constraints to this runs extraction using this extractor. This method has the following arguments:
    - `constraints`: A list of `ConstraintExtractor` objects.
    - `max_documents`: The maximum number of documents to go through in the `TextLoader`.
    - `max_seq_per_document`: The maximum number of chunks to use from each document. Setting this to a reasonable number can prevent over-representation from a single very long document.
    - `conjunction`. When set to `True`, the extractor will reject the sequence unless there is at least one satisfying target from _every_ `ConstraintExtractor` object passed to `constraints`. Otherwise, it will only require a single constraint extractor to have a satisfying target.
- `get_constraints`: After running `extract()`, there can be a combinatorially large number of possible constraints. This method allows us to sample a subset of these. This method has the following arguments:
    - `total_examples`: The maximum number of examples to sample. Note that there may not be that many examples, and so the actual returned number may vary. There are also no guarantees that all results are unique, such as when two distinct source sequences induce the same constraint.
    - `conjunction`: When set to `True`, each returned example must be a conjunction of all the constraints passed to `constraints` during `extract()`. Otherwise, picks one satisfying constraint at random for each example.

Here is an example of how to use the extractor together with the constraints we have defined above:
```python
import dill

extractor.extract(
    constraints=[constr_1, constr_2],
    max_documents=300,
    max_seq_per_document=100,
    conjunction=True
)
results = extractor.get_constraints(100, conjunction=True)

# it is recommended to dill pickle results, as this will store the Constraint objects
with open("results.dill", "wb") as f:
    dill.dump(results, f)
```

To see examples of how we integrate everything, see `collie/examples/extract.py`.
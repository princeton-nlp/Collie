"""Datasource extractor instances"""

from ..extract_constraints import FullExtractor
from ..extractor_utils import TextChunker, ConstraintExtractor
from ..extract_constraints import FullExtractor
from ..gutenberg_extractor import GutenbergLoader, get_gutenberg_filter, get_gutenberg_postprocessor
from ..ccnews_extractor import CCNewsLoader, get_ccnews_filter
from ..wiki_extractor import WikiLoader, get_wiki_filter, get_wiki_postprocessor
from ..english_extractor import EnglishLoader


SOURCE_PASSAGE_EXTRACTORS = {
    "guten": FullExtractor(
        chunker = TextChunker(
            paragraph_delim="\n\n",
            postprocessor=get_gutenberg_postprocessor(),
            filter=get_gutenberg_filter(),
            randomize=True,
            chunk_by_passage=True,
        ),
        loader = GutenbergLoader(
            "data/gutenberg-dammit-files/gutenberg-metadata.json",
            randomize=True
        ),
        metadata_fields=("Title", "Author","gd-path")
    ),
    "ccnews": FullExtractor(
        chunker = TextChunker(
            paragraph_delim="\n",
            randomize=True,
            filter=get_ccnews_filter(),
            chunk_by_passage=True,
        ),
        loader = CCNewsLoader(
            cache_dir="./data",
            randomize=True
        ),
        metadata_fields=("index", "title")
    ),
    "wiki": FullExtractor(
        chunker = TextChunker(
            paragraph_delim="\n\n",
            randomize=True,
            filter=get_wiki_filter(),
            postprocessor=get_wiki_postprocessor(),
            chunk_by_passage=True,
        ),
        loader = WikiLoader(
            cache_dir="./data",
            randomize=True
        ),
        metadata_fields=("index", "title")
    )
}


SOURCE_PARA_EXTRACTORS = {
    "guten": FullExtractor(
        chunker = TextChunker(
            paragraph_delim="\n\n",
            postprocessor=get_gutenberg_postprocessor(),
            filter=get_gutenberg_filter(),
            randomize=True,
            chunk_by_sentence=False,
        ),
        loader = GutenbergLoader(
            "data/gutenberg-dammit-files/gutenberg-metadata.json",
            randomize=True
        ),
        metadata_fields=("Title", "Author","gd-path")
    ),
    "ccnews": FullExtractor(
        chunker = TextChunker(
            paragraph_delim="\n",
            randomize=True,
            filter=get_ccnews_filter(),
            chunk_by_sentence=False,
        ),
        loader = CCNewsLoader(
            cache_dir="./data",
            randomize=True
        ),
        metadata_fields=("index", "title")
    ),
    "wiki": FullExtractor(
        chunker = TextChunker(
            paragraph_delim="\n\n",
            randomize=True,
            filter=get_wiki_filter(),
            postprocessor=get_wiki_postprocessor(),
            chunk_by_sentence=False,
        ),
        loader = WikiLoader(
            cache_dir="./data",
            randomize=True
        ),
        metadata_fields=("index", "title")
    )
}


SOURCE_SENT_EXTRACTORS = {
    "guten": FullExtractor(
        chunker = TextChunker(
            paragraph_delim="\n\n",
            postprocessor=get_gutenberg_postprocessor(),
            filter=get_gutenberg_filter(),
            randomize=True,
            chunk_by_sentence=True,
        ),
        loader = GutenbergLoader(
            "data/gutenberg-dammit-files/gutenberg-metadata.json",
            randomize=True
        ),
        metadata_fields=("Title", "Author","gd-path")
    ),
    "ccnews": FullExtractor(
        chunker = TextChunker(
            paragraph_delim="\n",
            randomize=True,
            filter=get_ccnews_filter(),
            chunk_by_sentence=True,
        ),
        loader = CCNewsLoader(
            cache_dir="./data",
            randomize=True
        ),
        metadata_fields=("index", "title")
    ),
    "wiki": FullExtractor(
        chunker = TextChunker(
            paragraph_delim="\n\n",
            randomize=True,
            filter=get_wiki_filter(),
            postprocessor=get_wiki_postprocessor(),
            chunk_by_sentence=True,
        ),
        loader = WikiLoader(
            cache_dir="./data",
            randomize=True
        ),
        metadata_fields=("index", "title")
    )
}


SOURCE_WORD_EXTRACTORS = {
    "english": FullExtractor(
        chunker = TextChunker(
            paragraph_delim="\n",
            randomize=True,
        ),
        loader = EnglishLoader(
            cache_dir="./data",
            randomize=True
        ),
    )
}
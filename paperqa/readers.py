from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Literal, overload
from itertools import zip_longest

import pymupdf
import pymupdf4llm
import tiktoken
from html2text import __version__ as html2text_version
from html2text import html2text

from paperqa.types import ChunkMetadata, Doc, ParsedMetadata, ParsedText, Text
from paperqa.utils import ImpossibleParsingError
from paperqa.version import __version__ as pqa_version


def parse_document(path: Path) -> ParsedText:
    
    return ParsedText(content=text, metadata=metadata)


def chunk_ParsedText(
    parsed_text: ParsedText, doc: Doc, chunk_chars: int, overlap: int
) -> list[Text]:
    
    text_content = parsed_text.content

    if not parsed_text.content:
        raise ImpossibleParsingError(
            "No text was parsed from the document: either empty or corrupted."
        )
    
    if len(text_content) > chunk_chars:
        text_chunks = [text_content[i:i+chunk_chars] for i in range(0, len(text_content), chunk_chars)]
    else:
        text_content = [text_content]

    texts = [Text(text=chunk, name=f"{doc.docname} chunk {i+1}") for i, chunk in enumerate(text_content)]
  
    return texts


def parse_text(
    path: Path, html: bool = False, split_lines: bool = False, use_tiktoken: bool = True
) -> ParsedText:
    """Simple text splitter, can optionally use tiktoken, parse html, or split into newlines.

    Args:
        path: path to file.
        html: flag to use html2text library for parsing.
        split_lines: flag to split lines into a list.
        use_tiktoken: flag to use tiktoken library to encode text.
    """
    try:
        with path.open() as f:
            text = list(f) if split_lines else f.read()
    except UnicodeDecodeError:
        with path.open(encoding="utf-8", errors="ignore") as f:
            text = f.read()

    parsing_libraries: list[str] = ["tiktoken (cl100k_base)"] if use_tiktoken else []
    if html:
        if not isinstance(text, str):
            raise NotImplementedError(
                "HTML parsing is not yet set up to work with split_lines."
            )
        text = html2text(text)
        parsing_libraries.append(f"html2text ({html2text_version})")

    return ParsedText(
        content=text,
        metadata=ParsedMetadata(
            parsing_libraries=parsing_libraries,
            paperqa_version=pqa_version,
            total_parsed_text_length=(
                len(text) if isinstance(text, str) else sum(len(t) for t in text)
            ),
            parse_type="txt" if not html else "html",
        ),
    )


def chunk_text(
    parsed_text: ParsedText, doc: Doc, chunk_chars: int, overlap: int, use_tiktoken=True
) -> list[Text]:
    """Parse a document into chunks, based on tiktoken encoding.

    NOTE: We get some byte continuation errors.
    Currently ignored, but should explore more to make sure we
    don't miss anything.
    """
    texts: list[Text] = []
    enc = tiktoken.get_encoding("cl100k_base")
    split = []

    if not isinstance(parsed_text.content, str):
        raise NotImplementedError(
            f"ParsedText.content must be a `str`, not {type(parsed_text.content)}."
        )

    content = parsed_text.content if not use_tiktoken else parsed_text.encode_content()

    # convert from characters to chunks
    char_count = parsed_text.metadata.total_parsed_text_length  # e.g., 25,000
    token_count = len(content)  # e.g., 4,500
    chars_per_token = char_count / token_count  # e.g., 5.5
    chunk_tokens = chunk_chars / chars_per_token  # e.g., 3000 / 5.5 = 545
    overlap_tokens = overlap / chars_per_token  # e.g., 100 / 5.5 = 18
    chunk_count = ceil(token_count / chunk_tokens)  # e.g., 4500 / 545 = 9

    for i in range(chunk_count):
        split = content[
            max(int(i * chunk_tokens - overlap_tokens), 0) : int(
                (i + 1) * chunk_tokens + overlap_tokens
            )
        ]
        texts.append(
            Text(
                text=enc.decode(split) if use_tiktoken else split,
                name=f"{doc.docname} chunk {i + 1}",
                doc=doc,
            )
        )
    return texts


def chunk_code_text(
    parsed_text: ParsedText, doc: Doc, chunk_chars: int, overlap: int
) -> list[Text]:
    """Parse a document into chunks, based on line numbers (for code)."""
    split = ""
    texts: list[Text] = []
    last_line = 0

    if not isinstance(parsed_text.content, list):
        raise NotImplementedError(
            f"ParsedText.content must be a `list`, not {type(parsed_text.content)}."
        )

    for i, line in enumerate(parsed_text.content):
        split += line
        while len(split) > chunk_chars:
            texts.append(
                Text(
                    text=split[:chunk_chars],
                    name=f"{doc.docname} lines {last_line}-{i}",
                    doc=doc,
                )
            )
            split = split[chunk_chars - overlap :]
            last_line = i
    if len(split) > overlap or not texts:
        texts.append(
            Text(
                text=split[:chunk_chars],
                name=f"{doc.docname} lines {last_line}-{i}",
                doc=doc,
            )
        )
    return texts


@overload
def read_doc(
    path: Path,
    doc: Doc,
    parsed_text_only: Literal[False],
    include_metadata: Literal[False],
    chunk_chars: int = ...,
    overlap: int = ...,
) -> list[Text]: ...


@overload
def read_doc(
    path: Path,
    doc: Doc,
    parsed_text_only: Literal[False] = ...,
    include_metadata: Literal[False] = ...,
    chunk_chars: int = ...,
    overlap: int = ...,
) -> list[Text]: ...


@overload
def read_doc(
    path: Path,
    doc: Doc,
    parsed_text_only: Literal[True],
    include_metadata: bool = ...,
    chunk_chars: int = ...,
    overlap: int = ...,
) -> ParsedText: ...


@overload
def read_doc(
    path: Path,
    doc: Doc,
    parsed_text_only: Literal[False],
    include_metadata: Literal[True],
    chunk_chars: int = ...,
    overlap: int = ...,
) -> tuple[list[Text], ParsedMetadata]: ...


def read_doc(
    path: Path,
    doc: Doc,
    parsed_text_only: bool = False,
    include_metadata: bool = False,
    chunk_chars: int = 3000,
    overlap: int = 100,
) -> list[Text] | ParsedText | tuple[list[Text], ParsedMetadata]:
    """Parse a document and split into chunks.

    Optionally can include just the parsing as well as metadata about the parsing/chunking

    Args:
        path: local document path
        doc: object with document metadata
        chunk_chars: size of chunks
        overlap: size of overlap between chunks
        parsed_text_only: return parsed text without chunking
        include_metadata: return a tuple
    """
    try:
        parsed_text = ParsedText(path, chunk_chars, overlap, parsed_text_only)
    except ImpossibleParsingError as e:
        # this file type cannot be parsed by parsed_text, handle gracefully
        ...
    except ValueError as e:
        # probably error in overlap vs chunk_chars, handle gracefully
        ...
    except Exception as e:
        # other error?
        ...

    if parsed_text_only:
        return parsed_text.text

    if include_metadata:
        return parsed_text.chunked, parsed_text.metadata

    return parsed_text.chunked

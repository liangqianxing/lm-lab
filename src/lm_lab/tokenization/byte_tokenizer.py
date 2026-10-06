"""T1: implement UTF-8 byte tokenization; see docs/tokenizer.md.

The interface is ready; the two algorithms below are intentionally unfinished.
"""

from collections.abc import Sequence
from typing import Literal


class ByteTokenizer:
    """A tokenizer whose IDs are exactly the 256 possible byte values."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        """Return UTF-8 byte values, preserving all text including whitespace.

        An empty string produces an empty list. Do not use Unicode code points
        as IDs and do not normalize text. TODO(T1): implement this operation.
        """
        utf_8bytes=text.encode('utf-8')
        return list(utf_8bytes)

    def decode(
        self, token_ids: Sequence[int], *, errors: Literal["strict", "replace"] = "strict"
    ) -> str:
        """Join byte values, then decode the entire byte string as UTF-8.

        Unknown IDs raise ValueError. Invalid UTF-8 raises UnicodeDecodeError
        in strict mode; replace mode is explicit and potentially lossy.
        Unsupported error modes raise ValueError. TODO(T1): implement.
        """

        for id in token_ids:
            if not (0<=id <=255):
                raise ValueError(f"Unkown ID {id} in ByteTokenizer.decode; must be 0-255")
        if errors not in ["strict", "replace"]:
                raise ValueError(f"Unsupported error mode {errors} in ByteTokenizer.decode; must be 'strict' or 'replace'")
        byte_string=bytes(token_ids)
        return byte_string.decode('utf-8', errors=errors)
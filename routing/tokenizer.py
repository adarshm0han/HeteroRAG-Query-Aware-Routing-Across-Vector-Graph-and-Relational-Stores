"""Use the existing token-counting asset locally; fail instead of downloading it."""

import hashlib
import os
from pathlib import Path

from baseline.common import sha256

ASSET_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
ASSET_NAME = hashlib.sha1(ASSET_URL.encode()).hexdigest()
ASSET_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


def local_tokenizer(workspace):
    import tiktoken

    cache = Path(
        os.environ.get("TIKTOKEN_CACHE_DIR", workspace / ".local/tiktoken-cache")
    )
    asset = cache / ASSET_NAME
    if not asset.is_file() or sha256(asset) != ASSET_SHA256:
        raise ValueError(
            "The verified o200k tokenizer cache is missing; see routing/README.md setup"
        )
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    return tiktoken.get_encoding("o200k_base")

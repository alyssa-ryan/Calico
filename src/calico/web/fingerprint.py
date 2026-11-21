"""A short hash of the web API's schema, so the page can tell it is talking to another version.

The page is built with the fingerprint of the schema it was generated from; the
server reports its own in ``/api/health``. When they differ, the server is running
older (or newer) code than the page, and requests would fail with confusing
validation errors, so the page asks for a restart instead.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def api_fingerprint(schema: Mapping[str, Any]) -> str:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]

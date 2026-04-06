import hashlib
import json
import os

import numpy as np


MODULE_CACHE_VERSION = "v1"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _update_hash_array(hasher, array):
    arr = np.ascontiguousarray(array)
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(arr.tobytes())


def build_cache_key(parts):
    hasher = hashlib.sha256()
    hasher.update(MODULE_CACHE_VERSION.encode("utf-8"))
    for part in parts:
        if isinstance(part, np.ndarray):
            _update_hash_array(hasher, part)
        elif isinstance(part, (list, tuple)):
            hasher.update(json.dumps(list(part), sort_keys=True).encode("utf-8"))
        elif isinstance(part, dict):
            hasher.update(json.dumps(part, sort_keys=True).encode("utf-8"))
        else:
            hasher.update(repr(part).encode("utf-8"))
    return hasher.hexdigest()


def cache_file(cache_dir, prefix, cache_key):
    ensure_dir(cache_dir)
    return os.path.join(cache_dir, f"{prefix}_{cache_key}.npz")

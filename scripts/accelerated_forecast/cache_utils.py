import errno
import hashlib
import json
import os
import socket
import shutil
import time
import uuid

import numpy as np


MODULE_CACHE_VERSION = "v1"
CACHE_READY_FILENAME = ".cache_ready"
CACHE_LOCK_SUFFIX = ".lock"
LOCK_OWNER_FILENAME = "owner.json"
DEFAULT_LOCK_STALE_SECONDS = 7200
DEFAULT_LOCK_POLL_SECONDS = 0.1


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


def cache_ready_path(cache_root):
    return os.path.join(cache_root, CACHE_READY_FILENAME)


def cache_lock_path(cache_root):
    return f"{cache_root}{CACHE_LOCK_SUFFIX}"


def is_complete_cache_dir(cache_root, required_files=None):
    if not os.path.isdir(cache_root):
        return False
    if os.path.exists(cache_ready_path(cache_root)):
        if required_files is None:
            return True
        return all(os.path.exists(os.path.join(cache_root, name)) for name in required_files)
    if required_files is None:
        return False
    return all(os.path.exists(os.path.join(cache_root, name)) for name in required_files)


def cleanup_incomplete_cache_dir(cache_root, required_files=None):
    if os.path.isdir(cache_root) and not is_complete_cache_dir(cache_root, required_files):
        shutil.rmtree(cache_root, ignore_errors=True)


def _temp_path(path):
    return f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"


def _write_npy(path, value):
    with open(path, "wb") as handle:
        np.save(handle, value, allow_pickle=False)


def _write_npz(path, value):
    with open(path, "wb") as handle:
        np.savez(handle, value)


def atomic_save_npy(path, value):
    ensure_dir(os.path.dirname(path))
    temp_path = _temp_path(path)
    try:
        _write_npy(temp_path, value)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def atomic_save_npz(path, value):
    ensure_dir(os.path.dirname(path))
    temp_path = _temp_path(path)
    try:
        _write_npz(temp_path, value)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _lock_owner_path(lock_path):
    return os.path.join(lock_path, LOCK_OWNER_FILENAME)


def _write_lock_owner(lock_path):
    with open(_lock_owner_path(lock_path), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at": time.time(),
            },
            handle,
        )


def _clear_stale_lock(lock_path, stale_seconds):
    if not os.path.isdir(lock_path):
        return False
    try:
        lock_age = time.time() - os.path.getmtime(lock_path)
    except OSError:
        return False
    if lock_age < stale_seconds:
        return False
    shutil.rmtree(lock_path, ignore_errors=True)
    return True


def build_cache_dir(
    cache_root,
    writer,
    required_files=None,
    stale_lock_seconds=DEFAULT_LOCK_STALE_SECONDS,
    poll_seconds=DEFAULT_LOCK_POLL_SECONDS,
):
    if is_complete_cache_dir(cache_root, required_files):
        return False

    ensure_dir(os.path.dirname(cache_root))
    lock_path = cache_lock_path(cache_root)

    while True:
        if is_complete_cache_dir(cache_root, required_files):
            return False
        try:
            os.mkdir(lock_path)
            _write_lock_owner(lock_path)
            break
        except FileExistsError:
            if _clear_stale_lock(lock_path, stale_lock_seconds):
                continue
            time.sleep(poll_seconds)

    try:
        if is_complete_cache_dir(cache_root, required_files):
            return False

        cleanup_incomplete_cache_dir(cache_root, required_files)
        temp_root = _temp_path(cache_root)
        os.makedirs(temp_root)
        try:
            writer(temp_root)
            with open(cache_ready_path(temp_root), "w", encoding="utf-8") as handle:
                handle.write("ok\n")
            try:
                os.replace(temp_root, cache_root)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not is_complete_cache_dir(
                    cache_root, required_files
                ):
                    raise
        finally:
            if os.path.isdir(temp_root):
                shutil.rmtree(temp_root, ignore_errors=True)
    finally:
        if os.path.isdir(lock_path):
            shutil.rmtree(lock_path, ignore_errors=True)

    return True


def publish_cache_dir(cache_root, writer, required_files=None):
    if is_complete_cache_dir(cache_root, required_files):
        return

    cleanup_incomplete_cache_dir(cache_root, required_files)
    ensure_dir(os.path.dirname(cache_root))
    temp_root = _temp_path(cache_root)
    os.makedirs(temp_root)
    try:
        writer(temp_root)
        with open(cache_ready_path(temp_root), "w", encoding="utf-8") as handle:
            handle.write("ok\n")
        try:
            os.replace(temp_root, cache_root)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not is_complete_cache_dir(
                cache_root, required_files
            ):
                raise
    finally:
        if os.path.isdir(temp_root):
            shutil.rmtree(temp_root, ignore_errors=True)

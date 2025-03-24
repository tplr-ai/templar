# ruff: noqa

print("tplr.patch_fs_cache loaded!")

import s3fs
from fsspec.spec import AbstractFileSystem


class DummyCache:
    def _log_stats(self, *args, **kwargs):
        # No-op: return empty stats.
        return {}

    def __getattr__(self, attr):
        # Return a dummy lambda for any attribute access.
        return lambda *args, **kwargs: None


def _get_cache(self):
    # Always return a DummyCache if no real cache is set.
    if getattr(self, "_cache", None) is None:
        return DummyCache()
    return self._cache


# Patch the AbstractFileSystem cache property.
setattr(AbstractFileSystem, "cache", {})  # type: ignore[attr-defined]

# Additionally, patch s3fs.S3FileSystem to ensure _cache is set properly.
_original_init = s3fs.S3FileSystem.__init__


def patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    if getattr(self, "_cache", None) is None:
        self._cache = DummyCache()


s3fs.S3FileSystem.__init__ = patched_init

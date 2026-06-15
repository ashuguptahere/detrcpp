# models/ — original-author checkpoints

detrcpp loads the **authors' own native `.pth`** from each model's **original upstream
repo** — never a Hugging Face mirror or port. Each model declares an `UpstreamRemapper()`
that maps the upstream parameter names onto our module tree, so the original checkpoint
loads **1:1 (0 missing / 0 unexpected / 0 shape-mismatched)** with no Python conversion.

This directory holds those downloaded checkpoints. The weights themselves are
git-ignored (see `.gitignore`); only this README is tracked.

## Downloading

The downloader is a pure-CMake script (no Python, no package manager):

```sh
cmake -P scripts/download_models.cmake -- list          # show the manifest
cmake -P scripts/download_models.cmake -- dfine-l        # one model
cmake -P scripts/download_models.cmake -- all            # every linked model
```

Files land here as `models/<filename>.pth`. Existing files are kept (delete to
re-fetch). Running a model whose weights are absent triggers an automatic download.

## Sources

Every entry points at the original upstream release (GitHub release assets, or the
authors' Google-Drive links). The manifest in `scripts/download_models.cmake` is the
single source of truth for URLs; `detrcpp --list-models` shows each model's provenance.

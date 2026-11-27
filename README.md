# Original inference repositories

This preservation branch holds the complete original files from the four local
inference folders reconciled into GH-INFERENCE on November 26, 2026. It is stored
inside the GH-INFERENCE GitHub repository so restoring the original work does not
depend on a separate local backup.

Use [main](https://github.com/TanishC4444/GH-INFERENCE) for maintained development.
Do not merge this preservation branch into the application branch.

## Included

`inference-originals.tar.gz` contains all 602 original files, including the full
`.git` directories, commit history, branches, uncommitted working files, ignored
files, experiment results, original packaging outputs, and caches. The four
archive folders are:

- `1-github-ai`: the desktop development copy with uncommitted vision support.
- `2-gh-ai-runner`: the older package copy originally named `gh-ai-runner:`.
- `3-gh-inference-source`: the older source/release snapshot.
- `4-gh-inference-fresh`: the standalone imported source snapshot.

The original folders remain unchanged. Reconciled source and experiment evidence
are also available as normal files on main. This branch additionally preserves
exact originals, older verification results, and Git metadata that main does not
need for normal development.

## Restore and verify

Download or clone this branch, then run:

```sh
python3 restore.py /path/to/a/new-empty-folder
```

The restore utility checks the archive checksum before extraction and checks
every extracted file against `manifest.json`. It requires a new, empty
destination and does not overwrite an existing project. Git histories can then
be inspected with `git -C /path/to/a/new-empty-folder/1-github-ai log`.

For a clone:

```sh
git clone --single-branch --branch preservation/inference-originals-2026-11-26 https://github.com/TanishC4444/GH-INFERENCE.git inference-originals
```

This preservation scope covers only these four inference folders. It does not
claim to preserve other projects or any future changes made after this snapshot.

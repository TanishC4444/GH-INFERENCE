# Original inference source and history

The original four inference working trees are browsable under [originals](originals/).
These are historical snapshots. Use the repository main branch for current code.

- [Desktop github-ai](originals/1-github-ai/)
- [Old gh-ai-runner](originals/2-gh-ai-runner/)
- [gh-inference-source](originals/3-gh-inference-source/)
- [gh-inference-fresh](originals/4-gh-inference-fresh/)

All original files outside .git, including ignored files and experiment results,
are stored directly as Git files. Original Git commit histories are incorporated
as parents of this branch's source-import commit, rather than nested .git folders.
The file-manifest.json records every original file checksum and all original
branch and tag identifiers. Git objects for each recorded history are reachable
from this branch. No archive or restore program is required to browse the work.

The prior archive commit remains in this branch's Git history, so even the exact
original Git metadata is recoverable if ever needed. Normal recovery requires
only cloning this branch. New local changes are not automatically included.

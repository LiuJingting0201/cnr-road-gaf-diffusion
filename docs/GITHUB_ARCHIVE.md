# GitHub deliverables and private reproducibility archive

Updated2026-11-09 under the user's request to push the project as comprehensively as possible, including large artifacts.

## Destinations

- Public source/results: [cnr-road-gaf-diffusion](https://github.com/LiuJingting0201/cnr-road-gaf-diffusion). Repository visibility was checked through GitHub on2026-11-09 and is **public**, contrary to earlier notes describing it as private. Its visibility is not changed by this backup.
- Private data/weights: [cnr-road-gaf-artifacts](https://github.com/LiuJingting0201/cnr-road-gaf-artifacts), created and independently checked as private. Access requires authorization to that repository.
- [Latest thesis PDF](../deliverables/thesis-v19/Road_Conditioned_Diffusion_Thesis_v19.pdf) and [Overleaf ZIP](../deliverables/thesis-v19/PoliTo_Thesis_v19_Overleaf.zip) are directly available in the public repository. Historical drafts are retained in the private archive.
- [G38 status](results/g38-route-preferences/STATUS.md) distinguishes completed route-model fits from the running generation-quality experiment. Archiving does not imply any new scientific result.

## Archive coverage

The initial cloud inventory covers10,023 selected files:63.87GB logical content,63.07GB after exact-content deduplication. This includes historical and current model checkpoints, prepared datasets, original simulator recordings, generated candidates, selection traces, source snapshots, configurations, logs, failure receipts and interrupted historical attempts. Early `data`/`runs` directories are included independently of their experiment-worktree symlinks. Original user-supplied CSVs and local deliverables are archived separately, preserving distinct file versions.

About8.36GB are excluded in the initial inventory: regenerable image caches and111 active-experiment snapshot files. Cache indices, encoder parameters, scripts and model weights remain. Python environments/installers, credential paths, system caches and incomplete local transfers are excluded. Full per-file exclusions and duplicate mappings belong to the private manifest. There were no credential-signature matches among selected small text files in the initial cloud inventory. Scientific target arrays are copied and hashed as opaque bytes, never decoded for fitting or evaluation by the backup.

Server files stream into512MiB archive parts; each uploaded part must match GitHub's returned SHA256 and size. At most a few temporary parts are staged on the server, with a16GiB minimum-free guard; original data are never deleted. Uploads use four concurrent network workers and low-priority archive compression. A token is used only for normal authenticated GitHub API access, retained in process memory and never written into files, command arguments, archives or logs.

## Status and recovery

The private Release uses tag `archive-2026-11-09`. It remains a draft until the planned assets and final manifest have been checked. **An upload queue being started is not evidence that all artifacts are already available.** Completion receipts and final coverage will be recorded here when verified.

Download the private Release assets with an authenticated GitHub client. Save `SERVER_MANIFEST.json` alongside the archive parts. Each group's parts are ordered by their four-digit suffix; concatenate them byte-for-byte to reconstruct its `.tar.gz`. Verify the part SHA256 values and the concatenated stream SHA256 in the manifest before extracting. Archives contain only regular files at relative paths rooted in `experiments/` and `legacy/`; do not extract outside a dedicated empty directory. After extracting unique files, use each manifest entry's `duplicate_of` mapping to restore duplicate paths from the canonical file and verify the final per-file SHA256.

Environment symlinks are inventory metadata, not instructions to follow external absolute paths. Recreate virtual environments/SUMO from the recorded requirements and version information, then map `experiments/data` and `experiments/runs` to the restored legacy data/runs directories. The source repository and per-job script snapshots preserve the implementations used by old experiments.

Large artifacts are Release attachments rather than ordinary Git blobs. [GitHub documents a2GiB maximum for each Release asset](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases); the512MiB parts stay below that boundary. No LFS paid-storage setting or account billing setting is changed.

## Verified recovery tooling

`scripts/archive_verify_restore.py` verifies per-part, full-stream and per-file SHA256 before or during restoration, reconstructs exact duplicate paths and preserves ordinary file modes. It rejects path traversal, non-regular tar members, corrupt content and existing restore destinations. The private Release includes a standalone copy and `RESTORE.md`. Full Git history is preserved as a private bundle rather than publishing all historical refs to the public code repository.

## Cloud closure checkpoint (2026-11-09)

All three cloud archive queues are now complete:118verified assets,53.08GiB,10,464distinct archived paths/72.29GBlogical cloud content. The supplemental archive restores all170initially excluded GAF image-cache files, the closed G38 snapshot and the old server working tree’s uncommitted source/logs plus its independent Git history. Initial active-file/cache exclusions above describe the first snapshot, not the final combined coverage. The old tree was never modified;314overlapping snapshot paths have identical hashes.

Actual GitHub download and full restoration of the current-model group succeeded:56logical/54unique files, all final file hashes rechecked, no weights deserialized or targets decoded. Local1,455file archival upload is still pending at this checkpoint; the Release stays draft until it is fully verified. Machine-readable evidence: `results/github-archive-20261109/cloud-complete.json` and `cloud-restore-checks.json`.

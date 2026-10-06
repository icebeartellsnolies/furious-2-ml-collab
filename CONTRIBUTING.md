# Contributing to Furious-2 ML Collaboration

## 🌿 Branching Model
All work must follow a one-way flow: `feat/` $\rightarrow$ `dev` $\rightarrow$ `staging` $\rightarrow$ `main`.

| Branch | Purpose | Created From | Merges Into |
|---|---|---|---|
| **main** | Production-ready, tagged models | — | — |
| **staging** | Release candidates for validation | main | main |
| **dev** | Integration of finished work | main | staging |
| **feat/** | Production code/pipeline changes | dev | dev |
| **data/** | Dataset updates (tracked via DVC) | dev | dev |
| **exp/** | Personal exploration/experiments | dev | Nothing (cherry-pick winner to feat/) |
| **fix/** | Urgent production bug fixes | main | main $\rightarrow$ dev |

**Rule:** No one pushes directly to `dev`, `staging`, or `main`. All changes must arrive via Pull Request (PR).

## ✍️ Commit Message Convention
We use [Conventional Commits](https://www.conventionalcommits.org/). Every commit must start with a type:

- `feat:` for new features or pipeline changes (e.g., `feat: add scaling step`)
- `data:` for dataset updates or DVC changes (e.g., `data: update training split`)
- `exp:` for experimental changes (e.g., `exp: try max_depth=10`)
- `fix:` for bug fixes (e.g., `fix: resolve path error in data_loader`)
- `docs:` for documentation changes

**Example:** `feat: implement random forest baseline`

## 🛠 Merge Strategy
**Decision: every PR is merged with a merge commit. We do not squash or rebase-merge.**
- **PRs into `dev`**: Merge commit. Each PR's commits stay on `dev` (e.g. `exp:` -> `feat:` history, and the `dvc.lock` regeneration commit), and every merge is traceable to its PR number.
- **Promotions (`dev` $\rightarrow$ `staging` $\rightarrow$ `main`)**: Merge commit, to preserve the release history.
- **Why not squash**: a squash would rewrite the commit that `metrics.json` logs as `commit_sha`, so the logged SHA would no longer exist on `dev`.
- Delete the source branch after merging (`feat/`, `data/`, `fix/`). Keep `exp/` branches.

## 🔁 Keeping branches up to date
Rebase your branch on `dev` (`git fetch && git rebase origin/dev`) before opening or updating a PR. If two PRs change the same line of `params.yaml`, the second author rebases, resolves the conflict, and documents it in the PR body.

## 🚀 Data & Model Workflow (DVC)
As this is an ML project, follow these rules:
1. **Always** run `dvc push` before `git push` when changing data or models.
2. Never commit raw `.csv`, `.pkl`, or `.joblib` files to Git.
3. Use `dvc pull` after cloning or switching branches to sync data.

### Lessons learned (gotchas)
- **Commit before you log.** `metrics.json` records `commit_sha`. Commit `params.yaml` and code first, then run `dvc repro -f -s evaluate`, so the logged SHA is a real commit on your branch. After `dvc exp apply`, a plain `dvc repro` can be served from the run cache and keep the old experiment's SHA.
- **Compare metrics, not hashes.** Windows CRLF can change the md5 in `dvc.lock` (e.g. for `src/features.py`) and even the `model.joblib` md5 while the metrics are identical. Reviewers compare `metrics.json` values.
- **`metrics.json` needs a trailing newline.** The `end-of-file-fixer` hook would otherwise change its md5 and make `dvc.lock` stale. Append a newline and run `dvc commit -f evaluate` before committing.
- **Reviewers run the pipeline.** For any PR that changes the pipeline, check out the branch, `dvc pull`, `dvc repro`, and confirm the metrics in the PR body. Reading the diff is not enough.
- **Short worktree paths on Windows.** `git worktree add` under a long path fails to remove ("Filename too long"); use a short path such as `C:\wt\pr`.

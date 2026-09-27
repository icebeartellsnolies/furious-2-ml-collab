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
To keep the project history readable:
- **PRs into `dev`**: We will use **Squash and Merge**. This condenses multiple "work-in-progress" commits into one clean feature commit.
- **Promotions (`dev` $\rightarrow$ `staging` $\rightarrow$ `main`)**: We will use **Merge Commits** to preserve the release history.

## 🚀 Data & Model Workflow (DVC)
As this is an ML project, follow these rules:
1. **Always** run `dvc push` before `git push` when changing data or models.
2. Never commit raw `.csv`, `.pkl`, or `.joblib` files to Git.
3. Use `dvc pull` after cloning or switching branches to sync data.

# Contributing Guidelines - Team Furious-2

Welcome to the **Furious-2 ML Collaboration** project. To ensure reproducibility, smooth teamwork, and compliance with the MLOps assignment rubric, all team members must follow these workflow rules.

---

## 1. Branching Strategy

Our repository uses a strict promotion workflow:
`feature / data / exp branches` $\longrightarrow$ `dev` $\longrightarrow$ `staging` $\longrightarrow$ `main`

| Branch | Purpose | Base Branch | Target Branch | Protection Rules |
| :--- | :--- | :--- | :--- | :--- |
| `main` | Production releases (tagged e.g. `model-v1.0`) | — | — | PR only, 1 approval, passing CI, no force push |
| `staging` | Release candidate validation (`dvc repro` reproduction) | `main` | `main` | PR only, 1 approval, passing CI |
| `dev` | Integration branch for completed work | `main` | `staging` | PR only, 1 approval, passing CI |
| `feat/<name>` | Production code, model features, pipeline stages | `dev` | `dev` | Delete branch after merge |
| `data/<name>` | Dataset updates tracked with DVC | `dev` | `dev` | Delete branch after merge |
| `exp/<member>-<idea>` | Exploration & experiments (e.g. `exp/bisma-rf-tuning`) | `dev` | *None* | Cherry-pick winner into `feat/` |
| `fix/<name>` | Urgent production bug fixes | `main` | `main` & `dev` | Delete branch after merge |

> [!IMPORTANT]
> **No Direct Pushes:** Nobody pushes directly to `dev`, `staging`, or `main`. All changes arrive through reviewed Pull Requests.

---

## 2. Commit Message Convention

We follow the **Conventional Commits** specification:
- `feat: <description>` - New features or pipeline additions (e.g. `feat: add scaling step`)
- `data: <description>` - Dataset updates or DVC tracking (e.g. `data: track raw dataset with DVC`)
- `exp: <description>` - Experimental configurations or explorations (e.g. `exp: try max_depth=10`)
- `fix: <description>` - Bug fixes (e.g. `fix: handle whitespace in column names`)
- `ci: <description>` - Continuous integration workflow changes
- `docs: <description>` - Documentation updates (`README.md`, `REPORT.md`)
- `chore: <description>` - Tooling, dependencies, or formatting updates

---

## 3. Pull Request & Merge Strategy

### Merge Decision
- **PRs into `dev`:** **Rebase-merge** (or squash-merge for multi-commit work-in-progress branches) to keep a clean, linear, and readable Git history.
- **PRs into `staging` & `main`:** Standard merge commits to preserve release milestones.

### Review Checklist Requirements
Every PR must fill out the repository template:
1. **No Data Leakage:** Preprocessing must only be fit on training splits.
2. **Fixed Seeds:** Seeds set for shuffling, initialization, and model training.
3. **No Hardcoded Paths:** Use relative paths or `pathlib.Path` rooted at project base.
4. **DVC Push Rule:** **Always run `dvc push` before `git push`** when data or model pointers change.
5. **Clean Notebooks:** Notebook outputs must be stripped (`nbstripout`) and paired with Jupytext.
6. **Linter & Tests Pass:** Pre-commit hooks and `pytest` must pass.

---

## 4. Environment & DVC Setup

1. Install project dependencies:
   ```bash
   uv sync
   ```
2. Pull tracked datasets and models:
   ```bash
   dvc pull
   ```
3. Run the pipeline:
   ```bash
   dvc repro
   ```

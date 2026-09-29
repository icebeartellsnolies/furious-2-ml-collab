# **Assignment01: Git-Based Collaboration for an ML Project** 

### **Overview** 

In teams of 2–3, you will take an existing ML dataset and training code, put it under Git and DVC, and run it like a real team project: branches, experiments, reviewed pull requests, versioned data, CI and a tagged release. The goal is the one from the course: **every result, reproducible by anyone on the team.** 

##### **What you will practise** 

- A clean repo layout, <mark>.gitignore</mark> and pre-commit hooks (Session 1) 

- A <mark>dev</mark> → <mark>staging</mark> → <mark>main</mark> workflow with feature and experiment branches and reviewed PRs (Session 2) 

- Notebooks that diff and merge cleanly with nbstripout and jupytext (Session 3) 

- Data and model versioning with DVC (Session 4) 

- Reproducible experiments: commit + config + data + environment + seed (Session 5) 

- CI checks on every PR and a reproducible release (Session 6) 

##### **Team rules** 

- 2–3 students per team. One repository per team, hosted on GitHub (or GitLab), with all members as collaborators and the instructor added as a viewer. 

- Every member must author commits, open PRs and review a teammate's PRs. Contribution is checked in the Git history, so do not commit on each other's behalf. 

- Nobody pushes directly to <mark>dev, staging</mark> or <mark>main</mark> . All changes arrive through pull requests. 

**What you submit:** the repository link, a <mark>REPORT.md</mark> in the repo, and a release tag <mark>model-v1.0</mark> on <mark>main</mark> that a stranger can reproduce. 

### **Branching model** 

Work flows in one direction: short-lived branches merge into <mark>dev</mark> , <mark>dev</mark> is promoted to <mark>staging,</mark> and <mark>staging</mark> is promoted to <mark>main.</mark> Only <mark>main</mark> represents production. 

|Branch|Purpose|Created from|Merges into|Protection|
|---|---|---|---|---|
|main|Production:<br>released, tagged<br>models only|—|—|PR only, 1<br>approval, all CI<br>checks pass|
|staging|Release<br>candidate,<br>reproduced and<br>validated before<br>release|main|main|PR only, 1<br>approval, all CI<br>checks pass|
|dev|Integration of|main|staging|PR only, 1|



|Branch|Purpose|Created from|Merges into|Protection|
|---|---|---|---|---|
||finished work|||approval, CI<br>passes|
|feat/<name>|Production code:<br>features, pipeline<br>changes|dev|dev|Deleted after<br>merge|
|data/<name>|Dataset updates<br>tracked with DVC|dev|dev|Deleted after<br>merge|
|exp/<member>-<br><idea>|Exploration; may<br>never merge|dev|Nothing directly:<br>cherry-pick the<br>winner into a<br>feat/ branch|Keep short-lived,<br>rebase ondev<br>often|
|fix/<name>|Urgent fix to<br>production|main|main,then back<br>into dev|Deleted after<br>merge|



The branch model has three permanent branches and four kinds of short-lived ones. Work always moves in one direction, from `dev` to `staging` to `main` , and only through reviewed pull requests with passing CI. 

- **`dev` (integration):** where finished work comes together. Every `feat/` and `data/` branch starts from `dev` and merges back into it through a PR that a teammate has reviewed. 

- **`staging` (release candidate):** when `dev` is ready, open a release PR into `staging` . A teammate who did not train the model clones the repo fresh, runs `dvc pull` and `dvc repro` , and confirms the metrics match exactly. 

- **`main` (production):** only reproduced, approved code from `staging` reaches `main` . Every merge into `main` is tagged, for example `model-v1.0` . 

- **`feat/<name>` :** production code changes, such as new features, pipeline steps or refactors. Keep these small and short-lived, and delete them after merging. 

- **`data/<name>` :** dataset changes tracked with DVC. Always run `dvc push` before `git push` , or your teammates will get broken data pointers. 

- **`exp/<member>-<idea>` :** your personal space for trying ideas. These branches are never merged directly. When an experiment wins, cherry-pick or re-apply the change onto a new `feat/` branch and open a PR from there. It's fine to abandon experiment branches. 

- **`fix/<name>` :** urgent fixes to production. Branch from `main` , merge back into `main` with a patch tag such as `model-v1.0.1` , then merge `main` back into `dev` so the fix isn't lost in the next release. 

Use <u>Conventional Commits for messages, for example</u> <mark>feat: add scaling step</mark> , <mark>data: remove duplicate rows, exp: try max_depth=8.</mark> As a team, decide once whether PRs into <mark>dev</mark> are squashmerged or rebase-merged, and write that decision in <mark>CONTRIBUTING.md.</mark> 

## **<u>Step-by-step tasks</u>** 

Complete the nine phases in order. Each phase ends with a checkpoint you must be able to show in the repository. 

#### **Phase 1 · Team and repository setup** 

1. Form a team of 2–3 and pick a team name. Choose a dataset and starter code from the list at the end of this document. 

2. One member creates an empty repository named <mark><team>-ml-collab</mark> and adds the teammates as collaborators with write access, plus the instructor. 

3. Every member clones it and sets their name and email: <mark>git config user.name</mark> and <mark>git config user.email</mark> . 

4. Assign roles for the project (each person owns one, but everyone codes and reviews): 

   - **Data owner:** DVC, data checks, dataset updates 

   - **Model owner:** training pipeline, configs, experiments 

   - **Platform owner:** CI, pre-commit, environment, releases (in a team of 2, split this role) 

**Checkpoint:** all members can push a branch to the repository. 

#### **Phase 2 · Scaffold the project and import the initial code** 

1. Create the standard layout (you may start from <u>Cookiecutter Data Science):</u> 

<mark>. ├── configs/          # params.yaml lives here or at the root ├── data/             # ignored by Git, tracked by DVC ├── models/           # ignored by Git, tracked by DVC ├── notebooks/ ├── src/              # reusable, tested code ├── tests/ ├── .github/workflows/ ├── .gitignore ├── .pre-commit-config.yaml</mark> 

- <mark>├── CONTRIBUTING.md</mark> 

- <mark>├── README.md └── pyproject.toml + uv.lock   (or requirements.txt with pinned versions)</mark> 

   2. Write a <mark>.gitignore</mark> that excludes datasets, checkpoints, <mark>.env, __pycache__/</mark> , <mark>.venv/</mark> and <mark>mlruns/</mark> . 

   3. Import the starter code into <mark>src/</mark> and refactor it just enough to run from the command line, for example <mark>python src/train.py</mark> . Remove every hardcoded absolute path. 

   4. Pin the environment: <mark>uv init</mark> + <mark>uv add scikit-learn pandas ...</mark> to produce <mark>uv.lock</mark> (or <mark>pip freeze > requirements.txt</mark> ). 

   5. Commit in small, well-described commits and push to <mark>main</mark> . This is the **only** time anyone pushes directly to <mark>main</mark> . 

   6. Create the long-lived branches and push them: 

<mark>git checkout -b staging && git push -u origin staging git checkout -b dev && git push -u origin dev</mark> 

7. Turn on branch protection for <mark>main, staging</mark> and <mark>dev</mark> : require a pull request, at least one 

approval, and (after Phase 8) passing status checks. Block force pushes. 

8. Add a <mark>CONTRIBUTING.md</mark> that states the branch naming rules, the commit message convention and your squash-vs-rebase decision. 

**Checkpoint:** three protected branches exist; <mark>git log</mark> on <mark>main</mark> shows the initial import. 

#### **Phase 3 · Guard rails: pre-commit and secrets** 

On a <mark>feat/pre-commit</mark> branch, add a <mark>.pre-commit-config.yaml</mark> with at least: <mark>ruff</mark> (lint + format), <mark>nbstripout</mark> , <mark>check-added-large-files</mark> (limit 1 MB) and a secret scanner such as <mark>detect-secrets</mark> or <mark>gitleaks</mark> . Every member runs <mark>pre-commit install</mark> . Open a PR into <mark>dev</mark> ; a teammate reviews and merges it. 

**Checkpoint:** committing a 5 MB file or a fake API key is blocked. Take a screenshot for the report. 

#### **Phase 4 · Version the data with DVC** 

On a <mark>data/initial-dataset</mark> branch (Data owner): 

<mark>uv add dvc          # add the extra for your remote, e.g. dvc[s3] or dvc[gdrive] dvc init dvc add data/raw/<dataset>.csv dvc remote add -d storage <your-remote>   # see Rules and tips git add data/raw/<dataset>.csv.dvc data/.gitignore .dvc/config git commit -m "data: track raw dataset with DVC" dvc push            # ALWAYS before git push git push -u origin data/initial-dataset</mark> 

Open a PR into <mark>dev</mark> . The reviewer must clone the branch fresh, run <mark>dvc pull,</mark> and confirm the data arrives. 

**Checkpoint:** the CSV is not in Git history; only its <mark>.dvc</mark> pointer is. 

#### **Phase 5 · Notebooks done right** 

On a <mark>feat/eda-notebook</mark> branch: 

1. Create <mark>notebooks/01-eda.ipynb</mark> exploring the dataset. 

2. Pair it with a script: <mark>jupytext --set-formats ipynb,py:percent notebooks/01-eda.ipynb</mark> and commit both files. 

3. Move one reusable function (for example, a cleaning or feature function) into <mark>src/</mark> with a unit test in <mark>tests/</mark> , then import it back into the notebook. 

4. Restart the kernel and run all cells before opening the PR. 

**Checkpoint:** the PR diff shows no cell outputs or execution counts. 

#### **Phase 6 · A reproducible pipeline** 

On a <mark>feat/dvc-pipeline</mark> branch (Model owner): 

1. Move every hyperparameter, split ratio and seed into <mark>params.yaml,</mark> for example: 

<mark>seed: 42 split: test_size: 0.2</mark> 

<mark>train:</mark> 

<mark>model: random_forest n_estimators: 100 max_depth: 6</mark> 

2. Split the code into <mark>prepare</mark> , <mark>train</mark> and <mark>evaluate</mark> stages and define them in <mark>dvc.yaml</mark> . <mark>evaluate</mark> writes <mark>metrics.json</mark> . 

3. Set seeds everywhere randomness occurs: splitting, shuffling, model initialisation, sampling. 

4. Fit preprocessing (scalers, encoders, imputers) on the training split only. 

5. Log the current commit SHA with every run (MLflow, W&B, or at minimum inside <mark>metrics.json)</mark> . 

6. Run <mark>dvc repro,</mark> then commit <mark>dvc.yaml</mark> , <mark>dvc.lock</mark> , <mark>params.yaml</mark> and <mark>metrics.json.</mark> Run <mark>dvc push,</mark> then <mark>git push</mark> . 

**Checkpoint:** a teammate on a fresh clone runs <mark>dvc pull && dvc repro</mark> and gets identical metrics. 

#### **Phase 7 · Experiments and pull requests** 

This phase is where most of the collaboration happens. 

1. **Experiments.** Each member creates their own branch from <mark>dev,</mark> such as <mark>exp/ali-max-depth,</mark> and runs at least **three** experiments, for example <mark>dvc exp run --set-param train.max_depth=10</mark> . Compare them with <mark>dvc exp show</mark> and paste the table into the PR description or report. Never run experiments on uncommitted code. 

2. **Promote the winner.** Apply the best experiment ( <mark>dvc exp apply <name>)</mark> , put the change on a <mark>feat/</mark> branch, and open a PR into <mark>dev</mark> with the metrics before and after. 

3. **Review each other.** Every PR is assigned to a teammate as reviewer. The reviewer fills in the review checklist (below) as a PR comment and requests changes at least once during the project. Each member must author at least **2 merged PRs** and review at least **2** . 

4. **Data update.** The Data owner opens a <mark>data/<change></mark> PR that modifies the dataset (remove duplicates, fix labels, add rows, or change the split). Show <mark>git checkout</mark> + <mark>dvc checkout</mark> moving between the old and new data version. 

5. **Resolve a real conflict.** Two members each change the same line of <mark>params.yaml</mark> on separate branches. Merge the first PR, then the second author rebases on <mark>dev</mark> , resolves the conflict, and documents the resolution in the PR. 

6. **Experiment drift.** Keep at least one <mark>exp/</mark> branch that is never merged, and explain in the report why it was abandoned. 

Add this template as <mark>.github/pull_request_template.md</mark> so every PR shows the checklist: 

<mark>## What changed and why</mark> 

<mark>## Metrics (before → after)</mark> 

<mark>## Review checklist</mark> 

- <mark>[ ] No data leakage (no target or future information in features) - [ ] Splits are fixed; preprocessing fit on training data only - [ ] No hardcoded paths; runs on a teammate's machine - [ ] Seeds set for shuffling, initialisation and sampling - [ ] Metric computed the way the team reports it</mark> 

- <mark>[ ] dvc push done before git push (if data or models changed)</mark> 

- <mark>[ ] Notebook restarted and run top to bottom (if notebooks changed)</mark> 

- <mark>[ ] Style and naming (linter passes)</mark> 

**Checkpoint:** the PR list shows every member as both author and reviewer, with at least one "changes requested" review. 

#### **Phase 8 · CI on every pull request** 

On a <mark>feat/ci</mark> branch (Platform owner), add a GitHub Actions workflow in <mark>.github/workflows/ci.yml</mark> that runs on every PR into <mark>dev, staging</mark> and <mark>main</mark> : 

1. **Lint:** <mark>ruff check</mark> and <mark>ruff format --check</mark> 

2. **Unit tests:** <mark>pytest tests/</mark> 

3. **Data checks:** schema, value ranges and null counts on the dataset or a committed sample 

4. **Smoke train:** train on a small sample (a few hundred rows) to prove the pipeline runs end to end 

5. **Bonus:** use CML to post the metrics table as a comment on the PR 

After it merges, make these checks required in the branch protection rules. 

**Checkpoint:** a deliberately broken test causes a red check that blocks merging. 

#### **Phase 9 · Release: dev → staging → main** 

1. Open a release PR from <mark>dev</mark> into <mark>staging</mark> titled <mark>release: v1.0</mark> that summarises the included PRs and the final metrics. 

2. **Reproducibility test.** A member who did not train the final model clones the repo into a new folder and runs: 

<mark>git clone <repo> && cd <repo> && git checkout staging uv sync dvc pull dvc repro</mark> 

They post the resulting <mark>metrics.json</mark> in the PR. The metrics must match the reported ones exactly. 

3. Merge into <mark>staging,</mark> then open a PR from <mark>staging</mark> into <mark>main.</mark> After approval and merge, tag the release: 

<mark>git checkout main && git pull git tag -a model-v1.0 -m "First production model" git push origin model-v1.0</mark> 

4. **Optional hotfix.** Find a small bug on <mark>main</mark> , fix it on a <mark>fix/</mark> branch, merge it into <mark>main,</mark> tag <mark>modelv1.0.1</mark> , and merge <mark>main</mark> back into <mark>dev.</mark> 

5. **Retrospective.** Meet as a team: what broke, what you would standardise, and what you added to <mark>CONTRIBUTING.md</mark> as a result. 

**Checkpoint:** <mark>model-v1.0</mark> exists on <mark>main</mark> and the reproduction in step 2 succeeded. 

### **Submission and grading** 

Submit the repository link. Grading is done from the repository alone, so everything below must be visible there. 

**<mark>REPORT.md</mark> must contain** 

- Team members, roles, dataset and starter-code source (with link) 

- A reproducibility table for the released model: commit SHA, <mark>params.yaml</mark> values, data <mark>.dvc</mark> hash, lock file, seed, and final metrics 

- The <mark>dvc exp show</mark> comparison of all experiments and why the winner was chosen 

- Links to: the data-update PR, the conflict-resolution PR, one "changes requested" review, the release PRs, and the abandoned <mark>exp/</mark> branch 

- Screenshots: blocked large file or secret, a failing CI check, a passing CI check 

- A short retrospective: what broke, and what you added to <mark>CONTRIBUTING.md</mark> because of it 

- One paragraph per member describing their own contribution 

##### **Rubric (100 points)** 

|Area|What is checked|Points|
|---|---|---|
|Repo structure and hygiene|Standard layout,.gitignore,<br>pre-commit hooks, no data or<br>secrets in Git history, clear<br>commits|10|
|Branching and protection|dev / staging /main protected;<br>correct branch names; no direct<br>pushes after Phase 2|15|
|Pull requests and review|Each member ≥ 2 authored and<br>≥ 2 reviewed PRs; checklist<br>used; at least one "changes<br>requested"; conflict resolved|20|
|Data and model versioning|DVC tracking and remote work;<br>data-update PR; old version<br>recoverable|15|
|Notebooks|Outputs stripped, jupytext<br>pairing, logic promoted to<br>testedsrc/|5|
|Reproducible experiments|params.yaml, dvc.yaml, seeds,<br>commit SHA logged, ≥ 3<br>experiments per member|15|
|CI|Lint, tests, data checks and<br>smoke train run on PRs and are<br>required|10|
|Release and report|Taggedmodel-v1.0,<br>independent reproduction<br>matches, complete REPORT.md|10|



|Area|What is checked|Points|
|---|---|---|
|**Total**||**100**|
|Bonus|CML metrics comment on PRs,|+5|
||or a completed hotfix with<br>model-v1.0.1||



Individual marks may be adjusted if the Git history shows a member did not author or review their share of the work. 

### **Datasets, starter code and tips** 

Pick a small tabular dataset (under about 50 MB) so pipelines and CI stay fast. Starter code can be your own earlier coursework, a scikit-learn example, or a public Kaggle notebook; credit the source in <mark>REPORT.md</mark> . 

|Dataset|Task|Source|
|---|---|---|
|Titanic|Binary classification|Kaggle|
|Wine Quality|Regression or classification|UCI ML Repository|
|Adult Income|Binary classification|UCI ML Repository|
|Heart Disease|Binary classification|UCI ML Repository|
|Telco Customer Churn|Binary classification|Kaggle|
|California Housing|Regression|sklearn.datasets.fetch_cal<br>ifornia_housing|



Each team should use a different dataset. Tell the instructor your choice by the end of Phase 1. 

**Choosing a DVC remote.** Teammates must be able to <mark>dvc pull,</mark> so the remote has to be shared. Good free options are a DagsHub repository (it provides a DVC remote), a shared Google Drive folder ( <mark>dvc[gdrive])</mark> , or an S3-compatible bucket if your institution provides one. Never commit remote credentials: keep them in <mark>.dvc/config.local</mark> or environment variables. 

##### **Common mistakes that cost marks** 

- Committing the dataset or a model file to Git. If it happens, remove it from history with <mark>git filter-repo,</mark> not just a new commit. 

- Running <mark>git push</mark> without <mark>dvc push</mark> , which leaves teammates with broken pointers. 

- Running experiments on uncommitted changes, so the logged SHA does not match the code. 

- Long-lived <mark>exp/</mark> branches that drift from <mark>dev.</mark> Rebase often or cherry-pick only what matters. 

- Notebooks that only work in one cell order on one laptop. 

- Approving a PR without actually running it. Reviewers should check out the branch at least once per PR that changes the pipeline. 

##### **Useful commands** 

<mark>git switch -c feat/<name> dev      # new branch from dev git fetch && git rebase origin/dev # keep your branch up to date git cherry-pick <sha>              # take one commit from an exp/ branch dvc status                         # what changed in data or pipeline dvc exp show                       # compare experiments dvc checkout                       # sync data to the current commit</mark> 


# Bisma Random Forest Depth Sweep

Branch: `exp/bisma-rf-sweep`

Baseline for DVC experiments: `dbc3e8c`

## DVC Experiments

| Experiment | Rev | max_depth | log_target | MAE | RMSE | R2 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `bisma-depth4` | `1e77857` | 4 | false | 2969.1332 | 11967.6409 | 0.0041 |
| `bisma-depth8` | `fa72e7e` | 8 | false | 2953.7986 | 11995.7040 | -0.0005 |
| `bisma-depth10` | `e11d11d` | 10 | false | 2982.0904 | 12031.9347 | -0.0066 |

## Dev Reproduction

After updating local `dev` to `origin/dev`, `dvc repro` produced:

| Branch | max_depth | log_target | MAE | RMSE | R2 |
| --- | ---: | --- | ---: | ---: | ---: |
| `dev` / `origin/dev` | 8 | true | 2303.8499 | 12015.5376 | -0.0038 |

`dev` performs better on MAE than the depth-only experiments.

## Remote Status

`dvc push` completed successfully after refreshing the local `pyOpenSSL` package used by the GDrive remote.

`dvc status -c` confirmed:

```text
Cache and remote 'storage' are in sync.
```

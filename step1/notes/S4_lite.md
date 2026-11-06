# S4 — lite policy training on this CPU (2026-11-06)

Preparation for Step 2; nothing goes into the paper. No install was needed.

**Commands**
- `./py scripts/lite/run_lite.py --name N --model released|rebuilt:A_s0 --num_envs 1024 --max_iterations 500 [--holdout 1,8] [--seed 0]`
- `./py scripts/lite/test_lite.py`: 10 checks, 11 s, all pass.
- Output goes to `runs/lite/N/`. The lite code is only read.

**Overrides** (no reward, observation, PPO setting or command range touched)
1. `device="cpu"`; `num_envs` and `max_iterations` are flags (defaults 8192, 500).
2. Dataset and checkpoint paths made absolute.
3. `torch.load` defaults to `map_location="cpu"`: the released checkpoint holds GPU tensors.
4. `TensorDictBase.__contains__` is `key in keys()`: tensordict 0.3.2 forbids `in`, which `resolve_obs_groups` uses.
5. `wandb` is a stub that writes the logged values to `metrics.jsonl`.
6. A `--seed` flag; upstream sets none.
7. `run_experiment` is repeated line for line (it needs a module-level `args_cli`).
8. For a rebuilt model, the random model the lite code builds is replaced through `adapter.py`.

**Adapter.** It calls the rebuild's heads as `SystemDynamicsEnsemble.forward` does, and adds `model_ids`, `reset_partial` and `architecture_config`. On 200 recorded windows it equals the rebuild exactly (limit 1e-6), for 1 and 5 heads. With `model_ids` and a partial reset it equals the upstream class with the released weights.

**Smoke runs** (10 iterations, 1,024 robots, 2 threads, with job S3 on the other cores): rewards and losses finite.
- Released: 3.41 s per iteration, 0.70 GB.
- A_s0: 2.78 s per iteration, 0.70 GB.

Ten iterations show no learning curve, so there is no claim about policy quality.

**Projection for 500 iterations** (steady seconds × 500; 4,096 robots timed over 4 iterations, 1.5 GB)

| robots | released | A_s0 |
|---|---|---|
| 1,024 | 0.47 h | 0.39 h |
| 4,096 | 1.88 h | 1.59 h |

Time grows linearly with robots. 8,192 would be about 3.7 h by extension (not measured). **Recommendation:** the CPU; a GPU notebook pays only for many policies, and uploading is the author's call.

**Step 2 requirements**
- **Start states.** The lite dataset holds windows from all 10 episodes. `--holdout 1,8` filters them in place after `prepare_data`, keeping a window only if every row is in a training episode (7,882 of 9,961). Use each model's own fold. Test 5 checks it.
- **Heads.** A 1-head rebuild has zero epistemic uncertainty, so the uncertainty penalty is inactive. Use 5-head rebuilds (`A5_*`) to keep it.

**Open.** Timings were taken under contention. With 1,024 robots PPO sees an eighth of the authors' samples per iteration. The `dof_acc_l2` quirk is unchanged.

# ems-dispatch-rl

Learning When Not to Send the Closest EMS Unit: a PPO ambulance dispatcher compared with
closest-unit dispatch. CS258 final project, UCR.

## Setup
```bash
pip install -r requirements.txt
```

## Files
| File | What | Owner |
|------|------|-------|
| `maps.py` | 16x16 city maps, travel times, named train/held-out maps | |
| `sim.py` | Simulator: calls, ambulances, events | |
| `env.py` | Gymnasium wrapper, observations, action masks | |
| `rewards.py` | Reward arms A–D (E2) | |
| `baselines.py` | Closest-unit baselines | |
| `train.py` | PPO + behavior cloning (E3) | |
| `evaluate.py` | Metrics and experiment runs (E1–E4) | |
| `plot.py` | Report figures and episode animations | |

Data goes in `data/` and outputs in `results/`; neither is committed.

## Shared interfaces
Change these only after telling the team; the rest of the code depends on them.

- **Maps** are looked up by name: `maps.get_map("train_00")`. Train on `TRAIN_MAPS`, keep
  `HELDOUT_MAPS` for E4.
- **Env** API, observation layout and `info` keys are documented at the top of `env.py`.
- **Policies** all have the signature `policy(env, obs) -> action`. Baselines are
  registered in `baselines.POLICIES`; `evaluate.load_policy` takes a baseline name or a
  PPO `.zip`, so every script can run any policy.
- **Rewards** are registered in `rewards.REWARDS` by arm letter.

## Running things
```bash
python env.py                                    # smoke test: one random episode
python maps.py                                   # draw all maps to results/maps/
python train.py --reward D --seed 0
python evaluate.py --policy closest_travel_time
python plot.py animate --policy results/ppo_D_seed0.zip --map heldout_03 --seed 7
```

## Workflow
Work on a branch (`git switch -c yourname/topic`), pull `main` often, and open a PR to merge.
Try to only edit the file(s) you own.

# ems-dispatch-rl

Learning When Not to Send the Closest EMS Unit: a PPO ambulance dispatcher compared with
closest-unit dispatch. CS258 final project, UCR.

**Read [CONTRACT.md](CONTRACT.md) first.** It holds the shared interfaces, conventions and
the open decisions (D1–D8) to settle at kickoff.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest          # unfinished parts skip themselves; nothing should fail
```

Clone the repo **outside Google Drive / iCloud / OneDrive**. Sync tools can corrupt `.git`.
Share code through GitHub and share data and results through the team Drive.

## Who owns what
Lanes from the team timeline. Put each person's name next to their M-number.

| Owner | Lane | Files |
|---|---|---|
| M1 | Simulator core | `sim.py`, `stub_sim.py` |
| M2 | Data and maps | `maps.py`, `calls.py`, `config.py`, `config/`, `fit_datasf.py`, `calibrate.py` |
| M3 | RL agent | `env.py`, `rewards.py`, `ppo_smdp.py`, `train.py` |
| M4 | Baselines, evaluation, BC | `baselines.py`, `evaluate.py`, `bc.py`, `plot.py` |
| All | Tests for your own files | `tests/` |

## Files
| File | What |
|---|---|
| `config/params.yaml` | Every shared constant (simulator, reward, loads, seeds). X values are placeholders |
| `config.py` | Loads `params.yaml` and `lambda_bar.json` |
| `maps.py` | 16×16 city maps, travel times, zones, named train/held-out maps |
| `calls.py` | The whole call stream for an episode, drawn at reset from the call seed |
| `sim.py` | Event-driven simulator, plus the `State`/`Reached` dataclasses everyone reads |
| `stub_sim.py` | Stand-in with the same interface, so env/baselines/eval can be built now |
| `env.py` | Gymnasium wrapper: observation layout, action masks, `info` keys |
| `rewards.py` | Reward arms A–D (E2) and how a step's rewards are combined |
| `ppo_smdp.py` | MaskablePPO with γ^Δt GAE and an actor/critic split (critic sees privileged values) |
| `bc.py` | Behavior cloning dataset, actor pretraining, critic warm-up, aux-loss schedule (E3) |
| `train.py` | Train one run: `--arm`, `--reward`, `--seed`, checkpoints and `--resume` |
| `baselines.py` | Closest-unit baselines and the random sanity check |
| `evaluate.py` | Runs any policy on fixed call seeds and writes a per-call CSV (E1–E4) |
| `plot.py` | Report figures and episode animations |
| `calibrate.py` | λ̄ per map and load → `config/lambda_bar.json` |
| `fit_datasf.py` | Fits the X values in `params.yaml` from DataSF |

## Running things
```bash
python -m pytest                                          # before every PR
python maps.py                                            # draw all maps to results/maps/
python calibrate.py                                       # after sim + maps work (T17)
python train.py --arm scratch --reward D --seed 0
python train.py --arm bc_ppo_aux --seed 0 --resume        # continue after a restart
python evaluate.py --policy closest_travel_time
python evaluate.py --policy results/scratch_D_seed0/model.zip --maps heldout --surge
python plot.py animate --policy results/scratch_D_seed0/model.zip --map heldout_03 --seed 100007
```

## Training on laptops
Set `--steps` from the profiling run (T18) so one run takes about 5 hours. Before an
overnight run, plug in and turn sleep off. Checkpoints save every 30 minutes, so a
stopped run continues with `--resume`.

## Workflow
Work on a branch (`git switch -c yourname/topic`), pull `main` often, and open a PR to merge.
Edit only the files you own, and tell the team before changing anything in CONTRACT.md.

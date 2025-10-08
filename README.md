# Experiments

The code is based on [ZSKT](https://github.com/polo5/ZeroShotKnowledgeTransfer).

## Run Experiments

General
* Single experiment. Distill from a `badnet_sq`-backoored teacher with arch WRN-16-2 to student with arch WRN-16-1.
    ```shell
    export CUDA_VISIBLE_DEVICES=0  # specify GPU
    python main.py --dataset=CIFAR10 --teacher_architecture=WRN-16-2 --student_architecture=WRN-16-1 --trigger_pattern=badnet_grid --seeds=3
    ```
* Run sweeps. Choose a sweep command below where you can find all hparams. For example,
    ```shell
    wandb sweep sweeps/cifar10_wrn_poi.yml
    # get the `wandb agent <agent code>` from the CLI output.
    wandb agent <agent code>  # this will run one pair of hyper-params from `cifar10_wrn_poi.yml`.
    ```
    `wandb agent <>` can be run in parallel in different processes, which will auto select different params in `yml` file.



## Distill from poisoned teachers

Evaluate different backdoors with ZSKT.
rename solver_zskt.py to solver.py and run the following:
```sh
python main_lf.py
```
Test backdoor target:
rename solver_bti.py to solver.py and run:
```sh
python main_lf.py
```
### Customization

**Add dataset**:
Edit `get_test_loader` in [zskt/datasets/datasets.py](zskt/datasets/datasets.py).

**Add model**:
Edit `zskt/models/selector.py` to add new architecture and pre-trained model paths.

## Defense

rename solver_cleandistill.py to solver.py and run:
```sh
python main_lf.py
```
from ray import tune
import click
import ray
import os
from random import randint


@click.group()
def cli():
    """A group of experiments for training Models
    """


@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--local-dir", type=str, default="RoMA/superconductor")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=4)
@click.option("--num-parallel", type=int, default=16)
@click.option("--num-samples", type=int, default=16)
def superconductor(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.safeweight import safeweight

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        safeweight,
        config={
            "logging_dir": "data",
            "task": "Superconductor-v0",
            "task_kwargs": {},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.2,
            "val_size": 500,
            "batch_size": 128,
            "updates": 500,
            "warmup_epochs": 50,
            "steps_per_update": 20,
            "hidden_size": 64,
            "model_lr": 0.001,
            "sol_x_samples": 128,
            "sol_x_lr": 3e-3,
            "coef_pessimism": 0,
            "coef_stddev": 0,
            "score_freq": 10,
            "model_class": "doublehead",
            "inner_lr": 5e-4,
            "region": 4,
            "alpha": 1.
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel},
    )

@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--local-dir", type=str, default="RoMA/dkitty")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=4)
@click.option("--num-parallel", type=int, default=16)
@click.option("--num-samples", type=int, default=16)
def dkitty(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.safeweight import safeweight

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        safeweight,
        config={
            "logging_dir": "data",
            "task": "DKittyMorphology-v0",
            "task_kwargs": {"split_percentile": 40, "num_parallel": 2},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.2,
            "val_size": 200,
            "batch_size": 128,
            "updates": 500,
            "warmup_epochs": 50,
            "steps_per_update": 20,
            "hidden_size": 64,
            "model_lr": 0.001,
            "sol_x_samples": 128,
            "sol_x_lr": 3e-3,
            "coef_pessimism": 0.0,
            "coef_stddev": 0.0,
            "score_freq": 10,
            "ema_rate": 0.0,
            "model_class": "doublehead",
            "inner_lr": 5e-3,
            "region": 4,
            "alpha": 1,
            "path": "./model/"

        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel},
    )


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--local-dir", type=str, default="RoMA/ant")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=4)
@click.option("--num-parallel", type=int, default=16)
@click.option("--num-samples", type=int, default=16)
def ant(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.safeweight import safeweight

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        safeweight,
        config={
            "logging_dir": "data",
            "task": "AntMorphology-v0",
            "task_kwargs": {"split_percentile": 20, "num_parallel": 2},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.2,
            "val_size": 200,
            "batch_size": 128,
            "updates": 500,
            "warmup_epochs": 100,
            "steps_per_update": 20,
            "hidden_size": 64,
            "model_lr": 0.001,
            "sol_x_samples": 128,
            "sol_x_lr": 3e-3,
            "coef_pessimism": 0.0,
            "coef_stddev": 0.0,
            "score_freq": 10,
            "ema_rate": 0.0,
            "model_class": "doublehead",
            "inner_lr": 5e-3,
            "region": 4,
            "alpha": 1,
            "path": "./model/"

        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel},
    )


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--local-dir", type=str, default="RoMA/hopper")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=4)
@click.option("--num-parallel", type=int, default=16)
@click.option("--num-samples", type=int, default=16)
def hopper(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.safeweight import safeweight

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        safeweight,
        config={
            "logging_dir": "data",
            "task": "HopperController-v0",
            "task_kwargs": {},
            "is_discrete": False,
            "normalize_ys": True,
            "normalize_xs": True,
            "continuous_noise_std": 0.2,
            "val_size": 200,
            "batch_size": 128,
            "updates": 500,
            "warmup_epochs": 200,
            "steps_per_update": 20,
            "hidden_size": 64,
            "model_lr": 0.001,
            "sol_x_samples": 128,
            "sol_x_lr": 2e-3,
            "coef_pessimism": 0.0,
            "coef_stddev": 0.0,
            "score_freq": 10,
            "ema_rate": 0.05,
            "model_class": "doublehead",
            "inner_lr": 5e-3,
            "region": 4.,
            "alpha": 1.,
            "path": "./model/"
        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel},
    )






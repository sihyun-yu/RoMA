from ray import tune
import click
import ray
import os
from random import randint


@click.group()
def cli():
    """A group of experiments for training
    """


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--local-dir", type=str, default="RoMA/gfp")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=4)
@click.option("--num-parallel", type=int, default=16)
@click.option("--num-samples", type=int, default=16)
def gfp(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.safeweight_latent import safeweight_latent

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        safeweight_latent,
        config={
            "logging_dir": "data",
            "task": "GFP-v0",
            "task_kwargs": {"seed": tune.randint(1000)},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "discrete_smoothing": 0.6,
            "continuous_noise_std": 0.2,
            "val_size": 200,
            "batch_size": 128,
            "updates": 500,
            "warmup_epochs": 100,
            "steps_per_update": 20,
            "hidden_size": 64,
            "model_lr": 1e-3,
            "sol_x_samples": 128,
            "sol_x_lr": 2e-3,
            "coef_pessimism": 0.0,
            "coef_stddev": 0.,
            "score_freq": 10,
            "model_class": "doublehead",
            "ema_rate": 0, 
            "inner_lr": 5e-4,
            "region": 4,
            "apply_mixup": False,

            "latent_size": 32,
            "vae_lr": 0.001,
            "vae_beta": 1.0,
            "offline_epochs": 200,
            "alpha": 1

        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel},
    )





@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--local-dir", type=str, default="RoMA/molecule")
@click.option("--cpus", type=int, default=24)
@click.option("--gpus", type=int, default=4)
@click.option("--num-parallel", type=int, default=16)
@click.option("--num-samples", type=int, default=16)
def molecule(local_dir, cpus, gpus, num_parallel, num_samples):
    from design_baselines.safeweight_latent import safeweight_latent

    ray.init(
        num_cpus=cpus, num_gpus=gpus, temp_dir=os.path.expanduser(f"~/tmp_{randint(0, 1000000)}"),
    )
    tune.run(
        safeweight_latent,
        config={
            "logging_dir": "data",
            "task": "MoleculeActivity-v0",
            "task_kwargs": {'split_percentile': 80},
            "is_discrete": True,
            "normalize_ys": True,
            "normalize_xs": False,
            "discrete_smoothing": 0.6,
            "continuous_noise_std": 0.2,
            "val_size": 200,
            "batch_size": 128,
            "updates": 500,
            "warmup_epochs": 100,
            "steps_per_update": 20,
            "hidden_size": 64,
            "model_lr": 1e-3,
            "sol_x_samples": 128,
            "sol_x_lr": 2e-3,
            "coef_pessimism": 0.0,
            "coef_stddev": 0.,
            "score_freq": 10,
            "model_class": "doublehead",
            "ema_rate": 0, 
            "inner_lr": 5e-4,
            "region": 4.,
            "apply_mixup": False,

            "latent_size": 32,
            "vae_lr": 0.001,
            "vae_beta": 1.0,
            "offline_epochs": 200,
            "alpha": 1

        },
        num_samples=num_samples,
        local_dir=local_dir,
        resources_per_trial={"cpu": cpus // num_parallel, "gpu": gpus / num_parallel},
    )




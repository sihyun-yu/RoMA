from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise, cont_noise, disc_noise
from design_baselines.safeweight.trainers import Trainer
from design_baselines.safeweight.nets import DoubleheadModel, SingleheadModel, NemoModel
from collections import defaultdict
import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfpd
import csv
import matplotlib.pyplot as plt


def normalize_dataset(x, y, normalize_xs, normalize_ys, is_discrete):
    if normalize_ys:
        # compute normalization statistics for the score
        mu_y = np.mean(y, axis=0, keepdims=True).astype(np.float32)
        y = y - mu_y
        st_y = np.std(y, axis=0, keepdims=True).astype(np.float32).clip(1e-6, 1e9)
        y = y / st_y

    else:
        # compute normalization statistics for the score
        mu_y = np.zeros_like(y[:1])
        st_y = np.ones_like(y[:1])

    if normalize_xs:
        # compute normalization statistics for the data vectors
        mu_x = np.mean(x, axis=0, keepdims=True).astype(np.float32)
        x = x - mu_x
        st_x = np.std(x, axis=0, keepdims=True).astype(np.float32).clip(1e-6, 1e9)
        st_x = np.where(np.equal(st_x, 0), 1, st_x)
        x = x / st_x


    else:
        # compute normalization statistics for the score
        mu_x = np.zeros_like(x[:1])
        st_x = np.ones_like(x[:1])

    return (x, mu_x, st_x), (y, mu_y, st_y)


def safeweight(config):
    # create the training task and logger
    logger = Logger(config["logging_dir"])
    task = StaticGraphTask(config["task"], **config["task_kwargs"])

    if config["is_discrete"]:
        task_x = task.x
        task_y = task.y        

        p = np.full_like(task_x, 1 / float(task_x.shape[-1]))
        discrete_clip = config.get('discrete_smoothing', 5.0)
        task_x = discrete_clip * task_x + (1.0 - discrete_clip) * p

        task_x = np.log(task_x)


    else:
        task_x = task.x
        task_y = task.y

    (task_x, mu_x, st_x), (task_y, mu_y, st_y) = normalize_dataset(
        x=task_x,
        y=task_y,
        normalize_xs=config["normalize_xs"],
        normalize_ys=config["normalize_ys"],
        is_discrete=config["is_discrete"]
    )

    input_shape = list(task.input_shape)


    max_y = np.amax(task_y)
    indices = tf.math.top_k(task_y[:, 0], k=config["sol_x_samples"])[1]

    sol_x =  tf.gather(task_x, indices, axis=0)
    sol_y = tf.gather(task_y, indices, axis=0)


    sol_x_opt = tf.keras.optimizers.Adam(learning_rate=config["sol_x_lr"])
    perturb_fn = lambda x: cont_noise(x, noise_std=config["continuous_noise_std"])

    model_class = {
        "doublehead": DoubleheadModel,
        "singlehead": SingleheadModel,
        "nemo": NemoModel,
    }.get(config["model_class"])

    model = model_class(
        input_shape=input_shape,
        hidden=config["hidden_size"],
    )
    temp_model = model_class(
        input_shape=input_shape,
        hidden=config["hidden_size"],
    )

    model_opt = tf.keras.optimizers.Adam(learning_rate=config["model_lr"])
    trainer = Trainer(
        model=model,
        model_opt=model_opt,
        perturb_fn=perturb_fn,
        is_discrete=config["is_discrete"],
        sol_x=sol_x,
        sol_y=sol_y,
        sol_x_opt=sol_x_opt,
        coef_stddev=config["coef_stddev"],
        temp_model=temp_model,
        steps_per_update=config["steps_per_update"],
        mu_x=mu_x,
        st_x=st_x,
        inner_lr=config["inner_lr"],
        region=config["region"],
        max_y=max_y,
        lr=config["sol_x_lr"],
        alpha=config["alpha"]
        )

    ### Warmup
    train_data, validate_data = task.build(
        x=task_x, y=task_y, batch_size=config["batch_size"], val_size=config["val_size"]
    )

    for epoch in range(config["warmup_epochs"]):
        statistics = defaultdict(list)
        for x, y in train_data:
            for name, tsr in trainer.train_step(x, y).items():
                statistics[f"warmup/train/{name}"].append(tsr)

        for x, y in validate_data:
            for name, tsr in trainer.validate_step(x, y).items():
                statistics[f"warmup/validate/{name}"].append(tsr)

        for name, tsrs in statistics.items():
            logger.record(name, tf.reduce_mean(tf.concat(tsrs, axis=0)), epoch)

    import json

    model.save('./model.pth')
    ### Solution Optimization
    step = 0
    update = 0
    while update < config["updates"]:
        trainer.init_step()
        train_statistics = defaultdict(list)
        statistics = trainer.fix_step()
        for name, tsr in statistics.items():
            logger.record(f"fix/{name}", tsr, update)

        statistics = trainer.update_step()
        for name, tsr in statistics.items():
            logger.record(f"update/{name}", tsr, update)

        if (update + 1) % config["score_freq"] == 0:
            sol_x = trainer.get_sol_x()
            inp = sol_x
            if config["is_discrete"]:
                solution = inp * st_x + mu_x
                score = task.score(tf.math.softmax(solution))
            else:
                score = task.score(inp * st_x + mu_x)


            logger.record(f"update/score", score, update, percentile=True, 
                molecule=(config["task"]=="MoleculeActivity-v0"))

        update += 1
        if update + 1 > config["updates"]:
            break










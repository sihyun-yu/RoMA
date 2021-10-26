from design_baselines.data import StaticGraphTask
from design_baselines.logger import Logger
from design_baselines.utils import spearman
from design_baselines.utils import soft_noise, cont_noise, disc_noise
from design_baselines.safeweight_latent.trainers import Trainer
from design_baselines.safeweight_latent.nets import DoubleheadModel, SingleheadModel, NemoModel
from collections import defaultdict
import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfpd

from design_baselines.cbas.nets import Encoder
from design_baselines.cbas.nets import DiscreteDecoder
from design_baselines.cbas.nets import ContinuousDecoder
from design_baselines.cbas.trainers import WeightedVAE
import os


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

def safeweight_latent(config):
    # create the training task and logger
    logger = Logger(config["logging_dir"])
    task = StaticGraphTask(config["task"], **config["task_kwargs"])

    task_x = task.x
    task_y = task.y        


    (task_x, mu_x, st_x), (task_y, mu_y, st_y) = normalize_dataset(
        x=task_x,
        y=task_y,
        normalize_xs=config["normalize_xs"],
        normalize_ys=config["normalize_ys"],
        is_discrete=config["is_discrete"]
    )

    # record the inputs shape of the forward model
    input_shape = list(task.input_shape)


    max_y = np.amax(task_y)
    indices = tf.math.top_k(task_y[:, 0], k=config["sol_x_samples"])[1]

    sol_x =  tf.gather(task_x, indices, axis=0)
    sol_y = tf.gather(task_y, indices, axis=0)


    ### Warmup
    train_data, validate_data = task.build(
        x=task_x, y=task_y, batch_size=config["batch_size"], val_size=config["val_size"]
    )


    sol_x_opt = tf.keras.optimizers.Adam(learning_rate=config["sol_x_lr"])
    perturb_fn = lambda x: cont_noise(x, noise_std=config["continuous_noise_std"])

    model_class = {
        "doublehead": DoubleheadModel,
        "singlehead": SingleheadModel,
        "nemo": NemoModel,
    }.get(config["model_class"])

    model = model_class(
        input_shape=[32],
        hidden=config["hidden_size"],
    )
    temp_model = model_class(
        input_shape=[32],
        hidden=config["hidden_size"],
    )

    decoder = DiscreteDecoder \
        if config['is_discrete'] else ContinuousDecoder

    # build the encoder and decoder distribution and the p model
    p_encoder = Encoder(task.input_shape,
                        config['latent_size'],
                        hidden=256)
    p_decoder = decoder(task.input_shape,
                        config['latent_size'],
                        hidden=256)
    p_vae = WeightedVAE(p_encoder,
                        p_decoder,
                        vae_optim=tf.keras.optimizers.Adam,
                        vae_lr=config['vae_lr'],
                        vae_beta=config['vae_beta'])


    # create a manager for saving algorithms state to the disk
    p_manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**p_vae.get_saveables()),
        os.path.join(config['logging_dir'], 'p_vae'), 1)


    # train the initial vae fit to the original data distribution
    p_manager.restore_or_initialize()
    p_vae.launch(train_data,
                 validate_data,
                 logger,
                 config['offline_epochs'])

    max_y = np.amax(task_y)
    indices = tf.math.top_k(task_y[:, 0], k=config["sol_x_samples"])[1]

    sol_x_opt = tf.keras.optimizers.Adam(learning_rate=config["sol_x_lr"])
    perturb_fn = lambda x: cont_noise(x, noise_std=config["continuous_noise_std"])

    #model_opt = tfa.optimizers.AdamW(learning_rate=config["model_lr"], weight_decay=config["wd"])
    model_opt = tf.keras.optimizers.Adam(learning_rate=config["model_lr"])

    task_d = p_encoder.get_distribution(task_x, training=False)
    task_x = task_d.sample().numpy()
    task_y = task.y

    for _ in range(9):
        np.append(task_x, task_d.sample().numpy())
        np.append(task_y, task.y)

    # renormalize
    (task_x, mu_x, st_x), (task_y, mu_y, st_y) = normalize_dataset(
        x=task_x,
        y=task_y,
        normalize_xs=True,
        normalize_ys=True,
        is_discrete=config["is_discrete"]
    )

    train_data, validate_data = task.build(
        x=task_x, y=task_y, batch_size=config["batch_size"], val_size=config["val_size"]
    )

    sol_x =  tf.gather(task_x, indices, axis=0)
    sol_y = tf.gather(task_y, indices, axis=0)

    trainer = Trainer(
        model=model,
        model_opt=model_opt,
        perturb_fn=perturb_fn,
        is_discrete=config["is_discrete"],
        sol_x=sol_x,
        sol_y=sol_y,
        sol_x_opt=sol_x_opt,
        coef_pessimism=config["coef_pessimism"],
        coef_stddev=config["coef_stddev"],
        temp_model=temp_model,
        steps_per_update=config["steps_per_update"],
        encoder=p_encoder,
        inner_lr=config["inner_lr"],
        region=config["region"],
        max_y=max_y,
        lr=config["sol_x_lr"],
        alpha=config["alpha"]
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

    model.save('./model/')
    ### Main training
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
            sol_z = trainer.get_sol_x()
            d = p_decoder.get_distribution(sol_z * st_x + mu_x, training=False)

            logits = d.logits
            maxlogit = tf.math.argmax(logits, axis=-1)
            maxlogit = tf.one_hot(maxlogit, input_shape[-1], axis=-1)
            score = task.score(maxlogit)
            
            logger.record(f"update/score", score, update, percentile=True, 
                molecule=(config["task"]=="MoleculeActivity-v0"))

            sample_score = task.score(d.sample())
            logger.record(f"update/sample_score", sample_score, update, percentile=True)

        update += 1
        if update + 1 > config["updates"]:
            break

 
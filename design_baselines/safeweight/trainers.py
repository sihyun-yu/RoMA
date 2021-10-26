from design_baselines.utils import spearman
from design_baselines.utils import perturb
import tensorflow.keras.layers as tfkl
import tensorflow as tf
import numpy as np
import random
import copy

class Trainer(tf.Module):
    def __init__(
        self,
        model,
        model_opt,
        perturb_fn,
        is_discrete,
        sol_x,
        sol_y,
        sol_x_opt,
        coef_stddev,
        temp_model,
        steps_per_update,
        mu_x,
        st_x,
        inner_lr=1e-3,
        region=2.,
        max_x = None,
        max_y = 1.,
        lr=1.,
        alpha=1.
    ):

        super().__init__()
        self.model = model
        self.model_opt = model_opt
        self.perturb_fn = perturb_fn
        self.is_discrete = is_discrete
        self.init_sol_x = sol_x
        self.sol_x = tf.Variable(sol_x)
        self.sol_x_opt = sol_x_opt
        self.coef_stddev = coef_stddev
        self.sol_x_samples = tf.shape(self.sol_x)[0]
        self.inner_lr = inner_lr
        self.temp_model = temp_model
        self.prev_sol_x = tf.Variable(sol_x)
        self.steps_per_update = steps_per_update
        self.beta = tf.Variable(0.1)
        self.region = region
        self.max_x = max_x
        self.max_y = max_y
        self.sol_y = tf.Variable(sol_y)
        self.temp_x = tf.Variable(sol_x)
        self.mu_x = mu_x
        self.st_x = st_x
        self.lr = lr
        self.alpha = alpha

    def get_sol_x(self):
        return self.sol_x.read_value()

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y):
        # copy the current model's weight into the temp model
        for weight, perturb in zip(self.model.trainable_variables, self.temp_model.trainable_variables):
            perturb.assign(weight)

        x = self.perturb_fn(x)

        for _ in range(self.steps_per_update ):
            with tf.GradientTape() as wp_tape:
                inp = x
                
                d = self.model.get_distribution(inp, training=True)
                temp_d = self.temp_model.get_distribution(inp, training=False)

                loss_pll = d.log_prob(y)                
                loss_total = tf.reduce_mean(loss_pll)
            grads = wp_tape.gradient(loss_total, self.model.trainable_variables)

            for weight, origin_weight, grad in zip(self.model.trainable_variables, self.temp_model.trainable_variables, grads):
                if tf.shape(grad)[0] > 1:
                    grad = tf.multiply(tf.norm(origin_weight)/tf.norm(grad), grad)
                    weight.assign(tf.subtract(weight, tf.multiply(self.inner_lr/self.steps_per_update, grad)))

        # calculate only 'perturbation'
        for weight, perturb in zip(self.model.trainable_variables, self.temp_model.trainable_variables):
            perturb.assign(weight - perturb)

        with tf.GradientTape() as outer_tape:
            inp = x
            d = self.model.get_distribution(inp, training=True)
            temp_d = self.temp_model.get_distribution(inp, training=False)

            loss_nll = -d.log_prob(y)

            rank_correlation = spearman(y[:, 0], d.mean()[:, 0])           
            loss_total = tf.reduce_mean(loss_nll)

        # take gradient steps on the model
        grads = outer_tape.gradient(loss_total, self.model.trainable_variables)
        grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
        self.model_opt.apply_gradients(zip(grads, self.model.trainable_variables))

        # substract the perturbation
        for weight, perturb in zip(self.model.trainable_variables, self.temp_model.trainable_variables):
            weight.assign(weight - perturb)


        statistics = dict()
        statistics["loss/total"] = loss_total
        statistics["rank_corr"] = rank_correlation

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def validate_step(self, x, y):

        x = self.perturb_fn(x)
        inp = x
        d = self.model.get_distribution(inp, training=True)
        loss_nll = -tf.reduce_mean(d.log_prob(y))
        rank_correlation = spearman(y[:, 0], d.mean()[:, 0])
        loss_total = loss_nll

        statistics = dict()
        statistics["loss/nll"] = loss_nll

        return statistics

    @tf.function(experimental_relax_shapes=True)
    def init_step(self):
        # copy the current model's weight into the old model
        for weight, perturb in zip(self.model.trainable_variables, self.temp_model.trainable_variables):
            perturb.assign(weight)

    @tf.function(experimental_relax_shapes=True)
    def fix_step(self):
        # weight perturbation (current, 1 step)
        for _ in range(self.steps_per_update * 5):
            with tf.GradientTape() as wp_tape:
                with tf.GradientTape(persistent=True) as inner_tape:
                    inner_tape.watch(self.sol_x)
                    inp = self.sol_x
                    prev_inp = self.init_sol_x
                    sol_d = self.temp_model.get_distribution(inp, training=True)
                    prev_sol_d = self.temp_model.get_distribution(prev_inp, training=True)
                    loss_sol_x = sol_d.mean() - self.coef_stddev * tf.math.log(sol_d.stddev())

                sol_x_grad = inner_tape.gradient(loss_sol_x, self.sol_x)
                loss_pessimism_gradnorm = tf.norm(tf.reshape(sol_x_grad, [self.sol_x_samples, -1]), axis=1)                
                loss_pessimism_con = -sol_d.log_prob(self.sol_y)
                loss_total = tf.reduce_mean(loss_pessimism_gradnorm) + self.alpha * tf.reduce_mean(loss_pessimism_con)


            grads = wp_tape.gradient(loss_total, self.temp_model.trainable_variables)

            for weight, origin_weight, grad in zip(self.temp_model.trainable_variables, self.model.trainable_variables, grads):
                if tf.shape(grad)[0] > 1:
                    grad = tf.multiply(tf.norm(origin_weight)/(tf.norm(grad)+1e-6), grad)
                    weight.assign(tf.subtract(weight, tf.multiply(self.inner_lr/self.steps_per_update/5., grad)))

        with tf.GradientTape() as wp_tape:
            inp = self.sol_x
            prev_inp = self.prev_sol_x
            sol_d = self.temp_model.get_distribution(inp, training=True)
            prev_sol_d = self.temp_model.get_distribution(prev_inp, training=True)

            
            loss_pessimism = (sol_d.mean() - self.coef_stddev * tf.math.log(sol_d.stddev())
                             - prev_sol_d.mean() - self.coef_stddev * tf.math.log(prev_sol_d.stddev())) ** 2
            loss_total = tf.reduce_mean(loss_pessimism)

        statistics = dict()
        statistics["loss"] = loss_total

        return statistics


    def update_step(self):
        statistics = dict()
        self.prev_sol_x.assign(self.sol_x)
        with tf.GradientTape() as tape:
            tape.watch(self.sol_x)
            inp = self.sol_x
            prev_inp = self.init_sol_x

            d = self.temp_model.get_distribution(inp, training=False)
            prev_sol_d = self.model.get_distribution(prev_inp, training=False)
            
            loss_pessimism = (d.mean() - self.coef_stddev * tf.math.log(d.stddev())
                             - prev_sol_d.mean() - self.coef_stddev * tf.math.log(prev_sol_d.stddev())) ** 2

            loss = (-(d.mean() - self.coef_stddev * tf.math.log(d.stddev()))
                    + (1./(2.*self.region)) * loss_pessimism
                    )

        sol_x_grad = tape.gradient(loss, self.sol_x)

        sol_x_grad_norm = tf.norm(tf.reshape(sol_x_grad, [self.sol_x_samples, -1]), axis=1)
        self.sol_x_opt.apply_gradients([[sol_x_grad, self.sol_x]])

        new_d = self.temp_model.get_distribution(self.sol_x, training=False)
        self.sol_y.assign(new_d.mean())


        travelled = tf.linalg.norm(self.sol_x - self.init_sol_x) / tf.cast(
            tf.shape(self.sol_x)[0], dtype=tf.float32
        )


        statistics["travelled"] = travelled

        return statistics

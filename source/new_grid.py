import tensorflow as tf
import numpy as np
import os
from dataclasses import dataclass
from sicore.intervals import intersection, union_all, not_
from sicore import tn_cdf_mpmath

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.config.experimental.enable_op_determinism()


def make_image(rng, d, signal, cov=1.0):
    a = d // 3
    cov = np.array(cov)
    if cov.shape == ():
        X = rng.normal(0, 1, (d, d, 1))
    else:
        X = rng.multivariate_normal(np.zeros(d * d), cov)
        X = np.reshape(X, (d, d, 1))
    abnormal_x = rng.integers(0, d - a)
    abnormal_y = rng.integers(0, d - a)
    X[abnormal_x : abnormal_x + a, abnormal_y : abnormal_y + a, 0] += signal
    return X


def create_cov_matrix(size, rho=0.5):
    matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(abs(i - j))
        matrix.append(row)
    cov = np.power(rho, matrix)
    return cov


@dataclass
class ResultOfInference:
    stat: float
    p_value: float
    inf_p: float
    sup_p: float
    truncated: list[list[float]]
    searched: list[list[float]]
    grid_size: int


class AdaptiveGridBasedSIforViT:
    def __init__(
        self, d, vit_visualizer, cov=1.0, threshold=0.6, eps_min=1e-4, eps_max=1e-1
    ):
        self.d = d
        self.vit_visualizer = vit_visualizer
        self.cov = tf.constant(cov, dtype=tf.float64)
        self.threshold = threshold
        self.grid_size = 0
        self.eps_min = eps_min
        self.eps_max = eps_max

    def construct_hypothesis(self, data):
        self.truncated = []
        self.searched = []

        data = tf.reshape(data, [1, self.d, self.d, 1])
        attention_region = self.vit_visualizer(data) >= self.threshold
        self.sign_map = tf.where(attention_region, 1.0, -1.0)
        data = tf.reshape(data, [-1])

        mask = tf.cast(tf.reshape(attention_region, [-1]), dtype=tf.float64)
        eta = (mask / tf.reduce_sum(mask)) - ((1.0 - mask) / tf.reduce_sum(1.0 - mask))

        if len(tf.shape(self.cov)) == 0:
            eta_norm = tf.norm(eta)
            self.stat = tf.tensordot(eta, data, axes=1) / (tf.sqrt(self.cov) * eta_norm)
            self.b = tf.sqrt(self.cov) * eta / eta_norm
            self.a = data - self.stat * self.b

        else:
            sigma_eta = tf.tensordot(self.cov, eta, axes=1)
            eta_sigma_eta = tf.tensordot(eta, sigma_eta, axes=1)
            sqrt_eta_sigma_eta = tf.sqrt(eta_sigma_eta)

            self.stat = tf.tensordot(eta, data, axes=1) / sqrt_eta_sigma_eta
            self.b = sigma_eta / sqrt_eta_sigma_eta
            self.a = data - self.stat * self.b

    def detect_cps(self):
        is_cp = tf.math.logical_xor(self.flags[:-1], self.flags[1:])
        loc_cp = tf.reshape(tf.where(is_cp), [-1])
        self.unsearched_flags = tf.reshape(tf.gather(self.flags, loc_cp), [-1])
        self.unsearched = tf.stack(
            [tf.gather(self.zs, loc_cp), tf.gather(self.zs, loc_cp + 1)], axis=1
        )

        n = tf.shape(self.zs)[0]
        loc_cp = tf.concat([[-1], loc_cp, [n - 1]], axis=0)
        flags = tf.reshape(tf.gather(self.flags, loc_cp[:-1] + 1), [-1])
        all_intervals = tf.stack(
            [tf.gather(self.zs, loc_cp[:-1] + 1), tf.gather(self.zs, loc_cp[1:])],
            axis=1,
        )
        not_empty = all_intervals[:, 1] - all_intervals[:, 0] >= 1e-12

        self.searched = all_intervals[not_empty].numpy().tolist()
        self.truncated = (
            all_intervals[tf.math.logical_and(flags, not_empty)].numpy().tolist()
        )

    def binary_search(self):
        for i in range(len(self.unsearched)):
            left_side_flag = self.unsearched_flags[i]
            left, right = self.unsearched[i]
            for _ in range(20):
                self.grid_size += 1
                mid = (left + right) / 2.0
                data = tf.reshape(self.a + self.b * mid, [1, self.d, self.d, 1])
                f_map = -(self.vit_visualizer(data) - self.threshold) * self.sign_map
                if tf.reduce_all(f_map <= 0.0):
                    if left_side_flag:
                        self.truncated.append([left.numpy(), mid.numpy()])
                        self.searched.append([left.numpy(), mid.numpy()])
                        left = mid
                    else:
                        self.truncated.append([mid.numpy(), right.numpy()])
                        self.searched.append([mid.numpy(), right.numpy()])
                        right = mid
                else:
                    if left_side_flag:
                        self.searched.append([mid.numpy(), right.numpy()])
                        right = mid
                    else:
                        self.searched.append([left.numpy(), mid.numpy()])
                        left = mid
                if right - left < 1e-10:
                    break

        truncated = union_all(self.truncated)
        searched = union_all(self.searched)
        self.truncated = truncated
        self.searched = searched

    def evaluate_pvalue(self):
        mask_intervals = [[-np.abs(self.stat.numpy()), np.abs(self.stat.numpy())]]
        inf_intervals = union_all(
            self.truncated + intersection(not_(self.searched), mask_intervals)
        )
        sup_intervals = union_all(
            self.truncated + intersection(not_(self.searched), not_(mask_intervals))
        )
        p_value = 1 - tn_cdf_mpmath(self.stat.numpy(), self.truncated, absolute=True)
        inf_p = 1 - tn_cdf_mpmath(self.stat.numpy(), inf_intervals, absolute=True)
        sup_p = 1 - tn_cdf_mpmath(self.stat.numpy(), sup_intervals, absolute=True)
        return inf_p, p_value, sup_p

    def inference(self):
        data = tf.reshape(self.a + self.b * self.stat, [1, self.d, self.d, 1])
        f_map = -(self.vit_visualizer(data) - self.threshold) * self.sign_map
        flag = tf.reduce_all(f_map <= 0.0)
        assert flag, f"stat is not in the region:{self.stat.numpy()}"

        left = -tf.abs(self.stat) - 10.0
        right = tf.abs(self.stat) + 10.0
        self.cut_grid(left, right)
        self.detect_cps()
        self.binary_search()
        inf_p, p_value, sup_p = self.evaluate_pvalue()

        return ResultOfInference(
            self.stat.numpy(),
            p_value,
            inf_p,
            sup_p,
            self.truncated,
            self.searched,
            self.grid_size,
        )

    @tf.function
    def evaluate_derivative(self, a, b, z, sign_map):
        with tf.autodiff.ForwardAccumulator(z, tf.ones_like(z)) as acc:
            data = a + b * z
            data = tf.reshape(data, [1, self.d, self.d, 1])
            weights = self.vit_visualizer.attention_rollout(data)
        jac1 = acc.jvp(weights)

        with tf.GradientTape() as tape:
            tape.watch(weights)
            raw_attention = self.vit_visualizer.upsample(weights)
        jac2 = tape.batch_jacobian(raw_attention, weights)

        grads = tf.einsum("bijk,blmnijk->blmn", jac1, jac2)
        grads = tf.cast(grads, tf.float32)

        with tf.autodiff.ForwardAccumulator(raw_attention, grads) as acc:
            attention = self.vit_visualizer.normalize(raw_attention)
            f_map = -(attention - self.threshold) * sign_map
        grads = acc.jvp(f_map)

        df = tf.reshape(grads, [-1])
        return df

    # confidence is not needed
    def cut_grid(self, left, right, is_set_stat=True):
        eps_min = tf.constant(self.eps_min, dtype=tf.float64)
        eps_max = tf.constant(self.eps_max, dtype=tf.float64)

        z_buffer = []
        flag_buffer = []
        f_buffer = []

        z = left
        while z < right:
            if is_set_stat and z > self.stat:
                z = self.stat
                is_set_stat = False

            data = tf.reshape(self.a + self.b * z, [1, self.d, self.d, 1])
            f_map = -(self.vit_visualizer(data) - self.threshold) * self.sign_map
            f = tf.reshape(f_map, [-1])
            f_buffer.append(f)
            flag = tf.reduce_all(f <= 0.0)

            if tf.abs(z - self.stat) < 1e-1:
                if flag:
                    step = tf.cast(tf.reduce_min(tf.abs(f / 1.0)), dtype=tf.float64)
                else:
                    step = tf.cast(tf.reduce_max(f / 1.0), dtype=tf.float64)
                step = tf.clip_by_value(step, eps_min, eps_max)
            else:
                df = self.evaluate_derivative(self.a, self.b, z, self.sign_map)

                conservative = -tf.cast(f, dtype=tf.float64) / (
                    10.0 * tf.cast(df, dtype=tf.float64)
                )
                conservative = tf.clip_by_value(conservative, -0.1, 2.0)  # too large
                conservative = tf.where(conservative < 0.0, 2.0, conservative)

                confidence = tf.where(
                    flag,
                    tf.reduce_min(conservative[f < 0.0]),
                    tf.reduce_max(conservative[f > 0.0]),
                )

                step = tf.clip_by_value(
                    tf.where(tf.math.is_nan(confidence), eps_min, confidence),
                    eps_min,
                    eps_max,
                )

            z_buffer.append(z)
            flag_buffer.append(flag)
            # print(z.numpy(), step.numpy(), flag.numpy())
            z += step

        self.zs = tf.reshape(z_buffer, [-1])
        self.flags = tf.reshape(flag_buffer, [-1])
        self.f = tf.reshape(f_buffer, [-1, self.d * self.d])

        self.grid_size += tf.shape(self.zs)[0]


class DeterministicGridBasedSIforViT:
    def __init__(self, d, vit_visualizer, cov=1.0, threshold=0.6, mode="fine"):
        self.d = d
        self.vit_visualizer = vit_visualizer
        self.cov = tf.constant(cov, dtype=tf.float64)
        self.threshold = threshold
        self.grid_size = 0
        self.mode = mode
        if self.mode == "combi":
            self.step = 0.01
        elif self.mode == "fine":
            self.step = 0.001
        else:
            raise ValueError("mode must be fine, coarse or combi")

    def construct_hypothesis(self, data):
        self.truncated = []
        self.searched = []

        data = tf.reshape(data, [1, self.d, self.d, 1])
        attention_region = self.vit_visualizer(data) >= self.threshold
        self.sign_map = tf.where(attention_region, 1.0, -1.0)
        data = tf.reshape(data, [-1])

        mask = tf.cast(tf.reshape(attention_region, [-1]), dtype=tf.float64)
        eta = (mask / tf.reduce_sum(mask)) - ((1.0 - mask) / tf.reduce_sum(1.0 - mask))

        if len(tf.shape(self.cov)) == 0:
            eta_norm = tf.norm(eta)
            self.stat = tf.tensordot(eta, data, axes=1) / (tf.sqrt(self.cov) * eta_norm)
            self.b = tf.sqrt(self.cov) * eta / eta_norm
            self.a = data - self.stat * self.b

        else:
            sigma_eta = tf.tensordot(self.cov, eta, axes=1)
            eta_sigma_eta = tf.tensordot(eta, sigma_eta, axes=1)
            sqrt_eta_sigma_eta = tf.sqrt(eta_sigma_eta)

            self.stat = tf.tensordot(eta, data, axes=1) / sqrt_eta_sigma_eta
            self.b = sigma_eta / sqrt_eta_sigma_eta
            self.a = data - self.stat * self.b

    def detect_cps(self):
        is_cp = tf.math.logical_xor(self.flags[:-1], self.flags[1:])
        loc_cp = tf.reshape(tf.where(is_cp), [-1])
        self.unsearched_flags = tf.reshape(tf.gather(self.flags, loc_cp), [-1])
        self.unsearched = tf.stack(
            [tf.gather(self.zs, loc_cp), tf.gather(self.zs, loc_cp + 1)], axis=1
        )

        n = tf.shape(self.zs)[0]
        loc_cp = tf.concat([[-1], loc_cp, [n - 1]], axis=0)
        flags = tf.reshape(tf.gather(self.flags, loc_cp[:-1] + 1), [-1])
        all_intervals = tf.stack(
            [tf.gather(self.zs, loc_cp[:-1] + 1), tf.gather(self.zs, loc_cp[1:])],
            axis=1,
        )
        not_empty = all_intervals[:, 1] - all_intervals[:, 0] >= 1e-12

        self.searched = all_intervals[not_empty].numpy().tolist()
        self.truncated = (
            all_intervals[tf.math.logical_and(flags, not_empty)].numpy().tolist()
        )

    def binary_search(self):
        for i in range(len(self.unsearched)):
            left_side_flag = self.unsearched_flags[i]
            left, right = self.unsearched[i]
            for _ in range(20):
                self.grid_size += 1
                mid = (left + right) / 2.0
                data = tf.reshape(self.a + self.b * mid, [1, self.d, self.d, 1])
                f_map = -(self.vit_visualizer(data) - self.threshold) * self.sign_map
                if tf.reduce_all(f_map <= 0.0):
                    if left_side_flag:
                        self.truncated.append([left.numpy(), mid.numpy()])
                        self.searched.append([left.numpy(), mid.numpy()])
                        left = mid
                    else:
                        self.truncated.append([mid.numpy(), right.numpy()])
                        self.searched.append([mid.numpy(), right.numpy()])
                        right = mid
                else:
                    if left_side_flag:
                        self.searched.append([mid.numpy(), right.numpy()])
                        right = mid
                    else:
                        self.searched.append([left.numpy(), mid.numpy()])
                        left = mid
                if right - left < 1e-10:
                    break

        truncated = union_all(self.truncated)
        searched = union_all(self.searched)
        self.truncated = truncated
        self.searched = searched

    def evaluate_pvalue(self):
        mask_intervals = [[-np.abs(self.stat.numpy()), np.abs(self.stat.numpy())]]
        inf_intervals = union_all(
            self.truncated + intersection(not_(self.searched), mask_intervals)
        )
        sup_intervals = union_all(
            self.truncated + intersection(not_(self.searched), not_(mask_intervals))
        )
        p_value = 1 - tn_cdf_mpmath(self.stat.numpy(), self.truncated, absolute=True)
        inf_p = 1 - tn_cdf_mpmath(self.stat.numpy(), inf_intervals, absolute=True)
        sup_p = 1 - tn_cdf_mpmath(self.stat.numpy(), sup_intervals, absolute=True)
        return inf_p, p_value, sup_p

    def inference(self):
        data = tf.reshape(self.a + self.b * self.stat, [1, self.d, self.d, 1])
        f_map = -(self.vit_visualizer(data) - self.threshold) * self.sign_map
        flag = tf.reduce_all(f_map <= 0.0)
        assert flag, f"stat is not in the region:{self.stat.numpy()}"

        left = -tf.abs(self.stat) - 10.0
        right = tf.abs(self.stat) + 10.0
        self.cut_grid(left, right)
        self.detect_cps()
        self.binary_search()
        inf_p, p_value, sup_p = self.evaluate_pvalue()

        return ResultOfInference(
            self.stat.numpy(),
            p_value,
            inf_p,
            sup_p,
            self.truncated,
            self.searched,
            self.grid_size,
        )

    # confidence is not needed
    def cut_grid(self, left, right, is_set_stat=True):
        z_buffer = []
        flag_buffer = []
        f_buffer = []

        z = left
        while z < right:
            if is_set_stat and z > self.stat:
                z = self.stat
                is_set_stat = False

            data = tf.reshape(self.a + self.b * z, [1, self.d, self.d, 1])
            f_map = -(self.vit_visualizer(data) - self.threshold) * self.sign_map
            f = tf.reshape(f_map, [-1])
            f_buffer.append(f)
            flag = tf.reduce_all(f <= 0.0)

            if self.mode == "combi" and tf.abs(z - self.stat) < 1e-1:
                step = 0.0001
            else:
                step = self.step

            z_buffer.append(z)
            flag_buffer.append(flag)
            # print(z.numpy(), step.numpy(), flag.numpy())
            z += step

        self.zs = tf.reshape(z_buffer, [-1])
        self.flags = tf.reshape(flag_buffer, [-1])
        self.f = tf.reshape(f_buffer, [-1, self.d * self.d])

        self.grid_size += tf.shape(self.zs)[0]

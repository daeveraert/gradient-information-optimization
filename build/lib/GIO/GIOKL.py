import jax.numpy as jnp
from jax import grad
import random
import jax

import numpy as np
import pyspark.sql.functions as F

from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from .GIO_super import GIO_super


class GIOKL(GIO_super):
    def __init__(self, uniform_low=-1, uniform_high=1, uniform_start_size=20, dim=768):
        super().__init__()
        self.spark = SparkSession.builder.getOrCreate()
        self.uniform_low = uniform_low
        self.uniform_high = uniform_high
        self.uniform_start_size = uniform_start_size
        self.dim = dim
        self.random_init = False

    def _get_nearest(self, sample, point):
        """Euclidean distance from point to it's nearest point in sample.
        :param sample: a set of points to compute the nearest distance to
        :param point: point to retrieve the nearest point in sample from
        :return: the index of the nearest point
        """
        norms = jnp.linalg.norm(sample - point, axis=1)
        return jnp.argsort(norms)[0]

    def _knn(self, x, y, k, last_only, discard_nearest, avg):
        """Find k_neighbors-nearest neighbor distances from y for each example in a minibatch x.
        :param x: tensor of shape [N_1, D]
        :param y: tensor of shape [N_2, D]
        :param k: the (k_neighbors+1):th nearest neighbor
        :param last_only: use only the last knn vs. all of them
        :param discard_nearest:
        :return: knn distances of shape [N, k_neighbors] or [N, 1] if last_only
        """

        dist_x = jnp.sum((x ** 2), axis=-1)[:, jnp.newaxis]
        dist_y = jnp.sum((y ** 2), axis=-1)[:, jnp.newaxis].T
        cross = - 2 * jnp.matmul(x, y.T)
        distmat = dist_x + cross + dist_y
        distmat = jnp.clip(distmat, 1e-10, 1e+20)

        if discard_nearest:
            if not avg:
                knn, _ = jax.lax.top_k(-distmat, k + 1)
            else:
                knn = -jnp.sort(distmat)
            knn = knn[:, 1:]
        else:
            knn = -distmat

        if last_only:
            knn = knn[:, -1:]

        return jnp.sqrt(-knn)

    def _kl_divergence_knn(self, x, y, k, eps, discard_nearest_for_xy):
        """KL divergence estimator for D(x~p || y~q).
        :param x: x~p
        :param y: y~q
        :param k: kth nearest neighbor
        :param discard_nearest_for_xy: discard nearest in the xy calculation
        :param eps: small epsilon to pass to log
        :return: scalar
        """
        n, d = x.shape
        m, _ = y.shape
        nns_xx = self._knn(x, x, k=k, last_only=True, discard_nearest=True, avg=False)
        nns_xy = self._knn(x, y, k=m, last_only=False, discard_nearest=discard_nearest_for_xy, avg=discard_nearest_for_xy)

        divergence = jnp.mean(d*jnp.log(nns_xy + eps) - d*jnp.log(nns_xx + eps)) + jnp.mean(jnp.log((k*m)/(jnp.arange(1, m+1) * (n-1))))

        return divergence

    def calculate_statistical_distance(self, x, y, k=5, eps=1e-8, discard_nearest_for_xy=False):
        """Calculate statistical distance d(p,q) based on x~p and y~q.
        :param x: x~p
        :param y: y~q
        :param k: kth nearest neighbor
        :param eps: small epsilon to pass to log
        :return: scalar
        """
        return self._kl_divergence_knn(x, y, k, eps, discard_nearest_for_xy)

    def gradient_descend(self, X, W, v, scaling_factor, max_iterations, lr=0.01, k=5, discard_nearest_for_xy=False):
        """Perform gradient descent on the statistical distance bwteen X and W+v
        :param X: target data
        :param W: current selected data
        :param v: initial v
        :param scaling_factor: scale the gradient
        :param max_iterations: iterations in the gradient descent
        :param lr: learning rate
        :param k: kth nearest neighbor
        :param discard_nearest_for_xy: discard nearest in the xy calculation
        :return: vector v opt
        """
        i = 0
        while i < max_iterations:
            gradient = grad(lambda v: self.calculate_statistical_distance(X, jnp.concatenate((W, v[jnp.newaxis, :])), k, discard_nearest_for_xy=discard_nearest_for_xy))(v)
            v = v - lr * scaling_factor * gradient
            i += 1
        return v

    def _get_uniform_start(self, do_normalize):
        """Get a uniform start for D.
        :return: jnp array of uniform points
        """
        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                return v
            return v / norm
        if do_normalize:
            return jnp.array([normalize(each) for each in np.random.uniform(low=self.uniform_low,high=self.uniform_high,size=(self.uniform_start_size,self.dim))])
        else:
            return jnp.array([each for each in np.random.uniform(low=self.uniform_low,high=self.uniform_high,size=(self.uniform_start_size,self.dim))])

    def fit(self, train, X, D=None, k=5, max_iter=100, stop_criterion="increase", min_difference=0, resets_allowed=False, max_resets=2, max_data_size=1, min_kl=0, max_sequential_increases=3, random_init_pct=0, random_restart_prob=0, scale_factor="auto", v_init='mean', grad_desc_iter=50, discard_nearest_for_xy=False, normalize=True, lr=0.01):
        """Perform GIO
        :param train: training data
        :param X: target data
        :param D: initial data
        :param k: kth nearest neighbor
        :param max_iter: max iterations for the algorithm
        :param stop_criterion: a string for the stopping criterion, one of the following:  'increase', 'max_resets', 'min_difference', 'sequential_increase_tolerance', 'min_kl', 'data_size'
        :param min_difference: the minimum difference between prior and current KL divergence for 'min_difference' stop criterion
        :param resets_allowed: whether if KL divergence increase, resetting G to the full train is allowed (allows the algorithm to pick duplicates). Must be set to true if the stop criterion is 'max_resets'
        :param max_resets: the number of resets allowed for the 'max_resets' stop criterion
        :param max_data_size: the maximum size of data for the 'data_size' stop criterion, as a percentage
        :param min_kl: the minimum kl divergence for the 'min_kl' stop criterion
        :param max_sequential_increases: the maximum number of sequential KL divergence increases for the 'sequential_increase_tolerance' stop criterion
        :param random_init_pct: the percent of training data to initialize the algorithm from
        :param random_restart_prob: probability to extend the gradient descent iterations by 3x to find potentially better extrema. Higher values come at the cost of efficiency
        :param scale_factor: factor to scale the gradient by or 'auto'
        :param v_init: how to initialize v in gradients descent, one of the following: 'mean', 'prev_opt', 'jump'
        :param grad_desc_iter: the number of iterations in gradient descent
        :param discard_nearest_for_xy: discard nearest in the xy calculation
        :param lr: Learning rate for gradient descent
        :return: selected data, kl divergences, (v, scale_factor, just_reset, num_resets, increases, adder, kl_divs)
        """
        if not random_init_pct and D is None:
            W = self._get_uniform_start(normalize)
            self.random_init = True
        elif D is None:
            amount = int(random_init_pct * len(train))
            W = jnp.array(random.sample(train.tolist(), amount))
        else:
            W = D[:]

        kl_dist_prev = self.calculate_statistical_distance(X, W, k, discard_nearest_for_xy=discard_nearest_for_xy)

        print("Starting KL: " + str(kl_dist_prev))
        if v_init == 'mean' or v_init == 'prev_opt':
            v = jnp.mean(X, axis=0)
        elif v_init == 'jump':
            v = jnp.array(random.sample(X.tolist(), 1)).squeeze()
        adder = train[:]
        kl_divs = []

        scale_factor = jnp.linalg.norm(v)/jnp.linalg.norm(grad(lambda v: self.calculate_statistical_distance(X, jnp.concatenate((W, v[jnp.newaxis, :])), k, discard_nearest_for_xy=discard_nearest_for_xy))(v)) if scale_factor == "auto" else scale_factor

        i = 0
        just_reset = False
        num_resets = 0
        total_iter = 0
        increases = 0
        while True:
            # Warmup, reset or random restart
            if i == 0 or just_reset or random.random() < random_restart_prob:
                v = self.gradient_descend(X, W, v, scale_factor, grad_desc_iter * 3, lr=lr, k=k)
            else:
                v = self.gradient_descend(X, W, v, scale_factor, grad_desc_iter, lr=lr, k=k)
            idx = self._get_nearest(v, adder)
            minvals = adder[idx]
            adder = jnp.delete(adder, idx, axis=0)

            W_tmp = jnp.concatenate((W, jnp.array(minvals)[jnp.newaxis, :]))

            kl_dist = self.calculate_statistical_distance(X, W_tmp, k, discard_nearest_for_xy=discard_nearest_for_xy)
            print("KL Divergence at iteration " + str(i) + ": " + str(kl_dist))

            # STOPPING CRITERIA
            if total_iter > max_iter:
                break

            if v_init == 'mean':
                v = jnp.mean(X, axis=0)
            elif v_init == 'jump':
                v = jnp.array(random.sample(X.tolist(), 1)).squeeze()

            adder, i, just_reset, stop, v, increases = self._test_stop_criterion(v_init, stop_criterion, kl_dist, kl_dist_prev, num_resets, max_resets, min_difference, increases, max_sequential_increases, min_kl, max_data_size, train, X, i, v, just_reset, resets_allowed, adder)

            if stop:
                break
            if not just_reset:
                W = W_tmp
                kl_divs += [kl_dist]
                kl_dist_prev = kl_dist
                i += 1
                total_iter += 1
        return W, kl_divs, (v, scale_factor, just_reset, num_resets, increases, adder, kl_divs)

    def _test_stop_criterion(self, v_init, stop_criterion, kl_dist, kl_dist_prev, num_resets, max_resets, min_difference, increases, max_sequential_increases, min_kl, max_data_size, train, X, i, v, just_reset, resets_allowed, adder):
        stop = False
        if stop_criterion == "increase" and kl_dist - kl_dist_prev > 0:
            stop = True
        elif stop_criterion == "max_resets" and kl_dist - kl_dist_prev > 0 and num_resets == max_resets:
            stop = True
        elif stop_criterion == "min_difference" and kl_dist_prev - kl_dist < min_difference:
            stop = True
        elif stop_criterion == 'sequential_increase_tolerance' and kl_dist - kl_dist_prev > 0 and increases == max_sequential_increases:
            stop = True
        elif stop_criterion == 'min_kl' and kl_dist < min_kl:
            stop = True
        elif stop_criterion == 'data_size' and i > int(max_data_size * len(train)):
            stop = True
        if stop:
            if just_reset:
                increases += 1
            if resets_allowed and num_resets < max_resets:
                num_resets += 1
                if v_init == 'prev_opt':
                    v = jnp.mean(X, axis=0)
                print("KL Div Increase, Resetting G")
                adder = train[:]
                i -= 1
                stop = False
            just_reset = True
        else:
            just_reset = False
            increases = 0
        return adder, i, just_reset, stop, v, increases

    def _return_kmeans(self, df, k, rseed):
        """Use Spark to perform K-Means
        :param df: dataframe to perform K-Means with
        :param k: number of clusters to compute
        :param rseed: random seed
        :return: k-means model, transformed df
        """
        kmeans = KMeans().setK(k).setSeed(rseed)
        model = kmeans.fit(df.select("features"))
        transformed_df = model.transform(df)
        return model, transformed_df

    def quantize(self, df_train, df_x, k=1500, rseed='auto', rseed1=234, rseed2=456):
        """Use Spark to perform K-Means
        :param df_train: train dataframe to quantize
        :param df_x: target dataframe to quantize
        :param k: number of clusters to compute
        :param rseed: 'auto' or 'manual'
        :param rseed1: first random seed
        :param rseed2: second random seed
        :return: k-means model, transformed df
        """
        if rseed == 'auto':
            rseed1 = random.randint(-1000,1000)
            rseed2 = random.randint(-1000,1000)
        model_train, transformed_train = self._return_kmeans(df_train, k, rseed1)
        model_X, transformed_X = self._return_kmeans(df_x, k, rseed2)
        return model_train, model_X, transformed_train, transformed_X

    def read_data_from_csv(self, path, path_X, delim="\t"):
        """Read in and process data stored in a csv. Data must be of the format: _c0, _c1, _c2 where _c2 contains the
        string representation of the vector, like "[0.1, 0.23, 0.45 ...]"
        :param path: path to training data
        :param path_X: path to target data
        :param delim: delimiter for csv file
        :return: train df, target df
        """
        new_schema = ArrayType(DoubleType(), containsNull=False)
        udf_json_to_arr = F.udf(lambda x: x, new_schema)

        df_read = self.spark.read.option("delimiter", delim).csv(path)
        df_with_embeddings = df_read.withColumn("features", udf_json_to_arr(F.from_json(F.col("_c2"), "array<double>")))

        df_X_read = self.spark.read.option("delimiter", delim).csv(path_X)
        df_X_with_embeddings = df_X_read.withColumn("features", udf_json_to_arr(F.from_json(F.col("_c2"), "array<double>")))

        return df_with_embeddings, df_X_with_embeddings

    def read_data_from_parquet(self, path, path_X):
        """Read in and process data stored in a parquet format. Data must contain a column "features" that stores an array<double>
        of the vectors and be non-nullable.
        :param path: path to training data
        :param path_X: path to target data
        :return: train df, target df
        """
        df_with_embeddings = self.spark.read.parquet(path)
        df_X_with_embeddings = self.spark.read.parquet(path_X)
        return df_with_embeddings, df_X_with_embeddings

    def explode(self, chosen_centroids, kmeans_transformed_df, kmeans_centroids_df):
        """Read in and process data stored in a parquet format. Data must contain a column "features" that stores an array<double>
        of the vectors.
        :param path: path to training data
        :param path_X: path to target data
        :return: train df, target df
        """
        pre_existing_centroids = jnp.array([f[1] for f in sorted([[each[0], each[1]] for each in kmeans_centroids_df.collect()], key=lambda x: x[0])])
        paired = []
        for each in chosen_centroids:
            for i, x in enumerate(pre_existing_centroids.tolist()):
                if each.tolist() in [x]:
                    paired += [i]
        print("Found " + str(len(paired)) + " centroids out of " + str(len(chosen_centroids)) + " selected centroids")
        full_selections_df = self.spark.createDataFrame(data=[(i, each) for i, each in enumerate(paired)], schema=["i", "id"]).join(kmeans_transformed_df, F.col("id") == F.col("prediction"))
        return full_selections_df


import jax.numpy as jnp
from jax import grad
import random
import jax

import numpy as np
import pyspark.sql.functions as F

from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.sql.types import *


class GIO_super:
    def __init__(self):
        pass

    def calculate_statistical_distance(self, x, y):
        pass

    def gradient_descend(self, X, W, v, factor, max_iterations, lr, *arg):
        pass

    def fit(self, train, X, *arg):
        pass

    def continue_fit(self, train, X, W, v_opt, just_reset, scale_factor, num_resets, increases, adder, kl_divs):
        pass

    def quantize(self, df_train, df_x, quantize_into):
        pass

    def _get_nearest(self, sample, point):
        pass

    def explode(self, chosen_centroids, kmeans_transformed_df, kmeans_centroids_df):
        pass

from GIO import GIOKL
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pyspark.sql.types import *
import pyspark.sql.functions as F

# Create some data
def getX():
    mean = [3,4]
    cov = [[0.5,0],[0,0.5]]
    np.random.seed(1)
    x, y = np.random.multivariate_normal(mean, cov, 100).T
    return jnp.array([[x[i],y[i]] for i in range(len(x))])

X = getX()

new_schema = ArrayType(DoubleType(), containsNull=False)
udf_no_null = F.udf(lambda x: x, new_schema)

# Initialize class
gio_kl = GIOKL.GIOKL(uniform_low=0, uniform_high=8, uniform_start_size=100, dim=2)
X_df = gio_kl.spark.createDataFrame(data=[(i, each.tolist()) for i, each in enumerate(X)], schema=["id", "features"]).withColumn("features", udf_no_null(F.col("features")))

# Quantize data
model_train, model_X, transformed_train, transformed_X = gio_kl.quantize(X_df, X_df)
quantized_X = jnp.array(model_X.clusterCenters())

# Calculate KL
kl = gio_kl.calculate_statistical_distance(X, quantized_X)

print("KL Divergence: " + str(kl))

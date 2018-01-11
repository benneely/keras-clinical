from keras.models import Model
import keras.applications
from keras.layers import Dense, Input
import numpy as np
from keras.applications import vgg16
from keras import backend as K


def GRACE_MORTALITY():
    """Instantiates the GRACE Mortality model (simple logistic regression architecture). Coefficients found from
    http://www.outcomes-umassmed.org/grace/files/GRACE_RiskModel_Coefficients.pdf

    # Arguments
        self: holding.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: Example
    """
    img_input = Input(shape=(8,))
    x = Dense(1, activation="sigmoid")(img_input)
    inputs = img_input
    model = Model(inputs, x, name='grace_mortality')
    betas = np.array([[0.0531, 0.0087, -0.0168, 0.1823, 0.6931, 1.4586, 0.4700, 0.8755]], dtype=np.float)
    intercept = np.array(([-7.7035]), dtype=np.float)
    weights = [betas.T, intercept]
    model.set_weights(weights)
    return model

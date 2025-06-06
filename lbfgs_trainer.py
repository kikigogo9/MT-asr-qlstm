from typing import Callable
import tensorflow as tf
import keras
import numpy as np
from scipy.optimize import minimize
import time

class Lbfgs:
    """Custom LBFGS trainer for tensorflow
    """
    loss_fn: Callable
    model: keras.layers.Layer
    dataset: tf.data.Dataset
    epoch: int
    max_iter: int
    
    def __init__(self, loss_fn: Callable, model: keras.layers.Layer, dataset: tf.data.Dataset, epoch: int = 5, max_iter: int = 10):
        self.loss_fn = loss_fn
        self.model = model
        self.dataset = dataset
        self.epoch = epoch
        self.max_iter = max_iter

    def get_weights_as_vector(self, model) -> np.ndarray:
        weights = model.get_weights()
        return np.concatenate([w.flatten() for w in weights])

    def set_weights_from_vector(self, model, weight_vector):
        weights = []
        start = 0
        for layer_weights in model.get_weights():
            shape = layer_weights.shape
            size = np.prod(shape)
            weights.append(weight_vector[start:start+size].reshape(shape))
            start += size
        model.set_weights(weights)

    def loss_and_gradients(self, model, weight_vector, X_batch, y_batch):
        self.set_weights_from_vector(model, weight_vector)
        with tf.GradientTape() as tape:
            predictions = model(X_batch, training=True)
            loss_value = self.loss_fn(y_batch, predictions)
        gradients = tape.gradient(loss_value, model.trainable_weights)
        grad_vector = np.concatenate([g.numpy().flatten() for g in gradients])
        return loss_value.numpy(), grad_vector * 10

    def train(self):
        num_batches = len(self.dataset)
        loss_history = []
        
        for epoch in range(self.epoch):
            epoch_time = time.time()
            print(f"Epoch {epoch + 1}/{self.epoch}")
            for batch_idx, (X_batch, y_batch) in enumerate(self.dataset):
                batch_time = time.time()


                # Optimize on the mini-batch using L-BFGS-B
                initial_params = self.get_weights_as_vector(self.model)
                result = minimize(
                    lambda w: self.loss_and_gradients(self.model, w, X_batch, y_batch),
                    initial_params,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 10}  # Adjust as needed
                )

                # Update model weights
                self.set_weights_from_vector(self.model, result.x)

                # Print progress
                print(f"Batch {batch_idx + 1}/{num_batches}, Loss: {result.fun} Batch time: {time.time()-batch_time}")
                loss_history.append(result.fun)
            print(f"Epoch time: {time.time()-epoch_time}")
        return loss_history
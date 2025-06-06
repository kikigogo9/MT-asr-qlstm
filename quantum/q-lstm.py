from ctypes import Array
from typing import Callable
import keras
import pennylane as qml
import numpy as np
import tensorflow as tf
from pennylane import CVNeuralNetLayers
from matplotlib import pyplot as plt
from tensorflow.keras import models


dev = qml.device('lightning.qubit', wires=4 )
#tf.autograd.set_detect_anomaly(True)

#tf.random.manual_seed(42)
dtype_global = tf.float32


class QLstm(keras.layers.Layer):
    layers: int
    wires: int
    featue_function_1: Callable
    featue_function_2: Callable

    
    def __init__(self,
                wires: int = 3,
                layers: int = 2,
                feature_dim: int = 1,
                name = 'qlstm',
                type='atan'
                ):
        super().__init__()

        self.wires = wires
        self.layers = layers
        self.name = name
        # We have 6 VQCs, therefore dimensionality -> (6, L, 3*W)
        # L - layers
        # W - number of wires

        self.phi = self.add_variable(
            name='ansatz',
            shape=(6, layers, 3*wires),
            initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.001),
        )
        
        self.feature_combiner = keras.layers.Dense(feature_dim + wires, dtype=dtype_global)
        
        self.__set_feature_function(type)
        
    
    def __set_feature_function(self, type='atan'):
        if type == 'atan':
            self.featue_function_1 = tf.atan
            self.featue_function_2 = lambda X: tf.atan(X**2)
        if type == 'asin':
            self.featue_function_1 = tf.asin
            self.featue_function_2 = tf.acos

    
    def layer(self, index, X: tf.Tensor):
        outputs = self.circuit(X, self.phi[index], self)
        return tf.stack(outputs).T
    
   
    @qml.qnode(dev, interface='tf', diff_method='adjoint')
    def circuit(X: tf.Tensor, weights: tf.Tensor, self):
        X_embedding_1 = self.featue_function_1(X)
        X_embedding_2 = self.featue_function_2(X)
        
        qml.AngleEmbedding(X_embedding_1, wires=range(self.wires), rotation="Y")
        qml.AngleEmbedding(X_embedding_2, wires=range(self.wires), rotation="Z")
        
        for layer in range(self.layers):
            # Entanglement
            for i in range(self.wires):
                qml.CNOT([i, (i+1) % self.wires])
                    # Entaglement every other one
            for i in range(self.wires):
                qml.CNOT([i, (i+2) % self.wires])
            for i in range(self.wires):        
                qml.RX(weights[layer, 3*i + 0], i)
                qml.RY(weights[layer, 3*i + 1], i)
                qml.RZ(weights[layer, 3*i + 2], i)
        

        return [qml.expval(qml.Z(i)) for i in range(self.wires)]

    def lstm(self, X: tf.Tensor, c_t_1):
        """

        Args:
            X (tf.Tensor): shape of (B, F)
            c_t_1 (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Collapse hidden state and input state onto the dimensions of the wires
        # Make sure the output of the sigmoid is in (-1, 1)
        X = 2.0 * self.sigmoid(self.feature_combiner(X)) - 1.0 
        
        f_t = tf.sigmoid(self.layer(index=0, X=X))
        i_t = tf.sigmoid(self.layer(index=1, X=X))
        C_t = tf.sigmoid(self.layer(index=2, X=X))
        o_t = tf.sigmoid(self.layer(index=3, X=X)) 

        c_t = f_t * c_t_1 + i_t * C_t
        
        rescaled = o_t*tf.tanh(c_t)
        
        h_t = self.layer(index=4, X=rescaled)
        y_t = self.layer(index=5, X=rescaled)
        
        self.c, self.h = c_t, h_t
        
        return y_t
    
    # input shoud be organized as (t, n)
    # t - timesteps
    # n - number of features
    # Return y_out (t, W) dimensions
    # t - timesteps
    # W - number of wires
    def call(self, input: tf.Tensor):
        # initialize hidden state
        batch_size = input.shape[0]
        self.h = tf.zeros((batch_size, self.wires), dtype=dtype_global)
        self.c =  tf.zeros((batch_size, self.wires), dtype=dtype_global)
        y_out = None
        for i in range(input.shape[1]): 
            # input -> (batch, time_steps, Features)
            y_t = self.lstm(tf.stack([input[:, i], self.h]), self.c)
            if y_out == None:
                y_out = y_t
            else:
                y_out = tf.concat([y_out, y_t], axis=1)
                
        y_out = tf.reshape(y_out, ( input.shape[0], input.shape[1], self.wires )) # B*t*W -> (B, t, W)
        return y_out, (self.c, self.h)
    
if __name__ == '__main__':
    qlstm = QLstm(4 , 2)
    
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    
    data = tf.range(0, 1, 0.01, tf.float32)
    data = tf.sin(data)
    dataset = []
    expected = []
    for i in range(data[:-8].shape[0]):
        dataset.append( tf.reshape(data[i:i+6],(6, 1)))
        expected.append([data[i+7]])
    dataset = tf.convert_to_tensor(dataset)
    expected = tf.convert_to_tensor(expected)
    
    loss = 0.0
    

    # for i in range(10):
    #     for j in range(expected.shape[0]):
    #         y_pred = qlstm.call(dataset[j])
    #         loss += loss(y_pred, expected[j])    

    #     optimizer.step()
    X_input = tf.keras.layers.Input((6, 1)) 
    model = models.Sequential([
        tf.keras.layers.Input((None, 6, 1), dtype=dtype_global),
        qlstm
    ])

    print(model.summary())

    plt.plot(expected)
    plt.plot(y_pred)
    plt.show
    
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import tensorflow as tf
tf.debugging.disable_traceback_filtering()
tf.get_logger().setLevel('ERROR')

import keras
import pennylane as qml
#os.environ["TF_USE_LEGACY_KERAS"] = "True"
import dev
import numpy as np

dtype_global = tf.float64

class QLSTM(keras.layers.Layer):
    def __init__(
            self,
            units: int,
            return_sequences: bool=True,
            wires: int = 3,
            layers: int = 2,
            feature_type='atan',
            dev = dev.dev,
            **kwargs
            ):
        super(QLSTM, self).__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.wires = wires
        self.layers = layers
        self.feature_type = feature_type
        self.dev = dev

        self.circuit = self.__get_circuit()
        shapes = {'weights': (self.layers, 3*self.wires)}
        self.shapes = shapes
        self.qlayers = [qml.qnn.KerasLayer(self.circuit, shapes, output_dim=self.wires),
            qml.qnn.KerasLayer(self.circuit, shapes, output_dim=self.wires),
            qml.qnn.KerasLayer(self.circuit, shapes, output_dim=self.wires),
            qml.qnn.KerasLayer(self.circuit, shapes, output_dim=self.wires),
            qml.qnn.KerasLayer(self.circuit, shapes, output_dim=self.wires),
            qml.qnn.KerasLayer(self.circuit, shapes, output_dim=self.wires)
            ]

        self.feature_combiner = keras.layers.Dense(self.wires, dtype=dtype_global)
        self.out_dense = keras.layers.Dense(self.units, dtype=dtype_global)
        self.__set_feature_function(self.feature_type)


    def build(self, input_shape):
        # Initialize Circuit embedding function
        super().build(input_shape)

    def __set_feature_function(self, feature_type='atan'):
        if feature_type == 'atan':
            self.featue_function_1 = tf.atan
            self.featue_function_2 = lambda X: tf.atan(X**2)
        if feature_type == 'asin':
            self.featue_function_1 = tf.asin
            self.featue_function_2 = tf.acos

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        # initialize hidden state
        batch_size = tf.shape(inputs)[0]

        self.h = tf.zeros((batch_size, self.wires), dtype=dtype_global)
        self.c =  tf.zeros((batch_size, self.wires), dtype=dtype_global)
        y_out = None
        for i in range(inputs.shape[1]):
            # input -> (batch, time_steps, Features)
            y_t = self.lstm(tf.concat([inputs[:, i], self.h], axis=1), self.c)
            if y_out is None:
                y_out = y_t
            else:
                y_out = tf.concat([y_out, y_t], axis=1)
                
        # B*t*W -> (B, t, W)
        y_out = tf.reshape(y_out, ( inputs.shape[0], self.wires, inputs.shape[1] ))
        y_out = tf.transpose(y_out, [0, 2, 1])
        y_out = self.out_dense(y_out)
        return y_out if self.return_sequences else y_out[:, -1, :], (self.c, self.h)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'return_sequences': self.return_sequences,
            'wires': self.wires,
            'layers': self.layers,
            'feature_type': self.feature_type
        })
        return config

    def layer(self, index, inputs: tf.Tensor) -> tf.Tensor:
        return tf.cast(self.qlayers[index](inputs), dtype=dtype_global)

    def __get_circuit(self) -> qml.QNode:
        def circuit(inputs: tf.Tensor, weights: tf.Tensor):
            x_embedding_1 = self.featue_function_1(inputs)
            x_embedding_2 = self.featue_function_2(inputs)
            qml.AngleEmbedding(x_embedding_1, wires=range(self.wires), rotation="Y")
            qml.AngleEmbedding(x_embedding_2, wires=range(self.wires), rotation="Z")
            for layer in range(self.layers):
                # Entanglement
                for i in range(self.wires):
                    qml.CNOT([i, (i+1) % self.wires])
                for i in range(self.wires):
                    qml.RX(weights[layer, 3*i + 0], i)
                    qml.RY(weights[layer, 3*i + 1], i)
                    qml.RZ(weights[layer, 3*i + 2], i)
            return [qml.expval(qml.Z(i)) for i in range(self.wires)]
        
        return qml.QNode(circuit, self.dev, dtype=dtype_global, diff_method='adjoint')
    
    def lstm(self, inputs: tf.Tensor, c_t_1) -> tf.Tensor:
        """

        Args:
            X (tf.Tensor): shape of (B, F)
            c_t_1 (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Collapse hidden state and input state onto the dimensions of the wires
        # Make sure the output of the sigmoid is in (-1, 1)
        inputs = self.feature_combiner(inputs)

        f_t = tf.sigmoid(self.layer(index=0, inputs=inputs))
        i_t = tf.sigmoid(self.layer(index=1, inputs=inputs))
        C_t = tf.tanh(self.layer(index=2, inputs=inputs))
        o_t = tf.sigmoid(self.layer(index=3, inputs=inputs))

        c_t = f_t * c_t_1 + i_t * C_t

        o = o_t*tf.tanh(c_t)

        h_t = self.layer(index=4, inputs=o)
        y_t = self.layer(index=5, inputs=o)

        self.c, self.h = c_t, h_t

        return y_t
   
    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, input_shape[-2], self.units]) if self.return_sequences else tf.TensorShape([None, self.units])

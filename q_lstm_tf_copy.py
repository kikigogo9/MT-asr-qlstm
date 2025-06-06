import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import tensorflow as tf
#tf.debugging.disable_traceback_filtering()
tf.get_logger().setLevel('ERROR')

import keras
import pennylane as qml
#os.environ["TF_USE_LEGACY_KERAS"] = "True"
import dev
import numpy as np

dtype_global = tf.float64

import tensorflow as tf
import pennylane as qml
import keras
import numpy as np

# ... (Keep your initial imports and environment setup)

class QLSTM(keras.layers.Layer):
    def __init__(
            self,
            units: int,
            return_sequences: bool = True,
            wires: int = 3,
            layers: int = 1,  # Reduced from 2 to mitigate barren plateaus
            feature_type='atan',
            **kwargs
    ):
        super(QLSTM, self).__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.wires = wires
        self.layers = layers
        self.feature_type = feature_type

        # Shared quantum layers for efficiency
        self.circuit = self.__get_circuit()
        shapes = {'weights': (self.layers, 3*self.wires)}
        
        # Use 4 quantum layers instead of 6 (standard LSTM gates)
        self.qlayers = qml.qnn.KerasLayer(self.circuit, shapes, output_dim=self.wires)

        self.feature_combiner = keras.layers.Dense(
            self.wires, 
            dtype=tf.float64
        )
        self.out_dense = keras.layers.Dense(self.units, dtype=tf.float64)

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        batch_size = tf.shape(inputs)[0]
        self.h = tf.zeros((batch_size, self.wires), dtype=tf.float64)
        self.c = tf.zeros((batch_size, self.wires), dtype=tf.float64)
        
        outputs = []
        for i in range(inputs.shape[1]):
            x = inputs[:, i]
            combined = tf.concat([x, self.h], axis=1)
            embedded = self.feature_combiner(combined)
            
            # Quantum operations with shared parameters
            q_output = self.qlayers(embedded)
            
            # Split quantum output for different gates
            f_t, i_t, c_hat_t, o_t = tf.split(q_output, 4, axis=1)
            
            # Corrected activation functions
            f_t = tf.sigmoid(f_t)
            i_t = tf.sigmoid(i_t)
            c_hat_t = tf.tanh(c_hat_t)  # Corrected from sigmoid to tanh
            o_t = tf.sigmoid(o_t)
            
            # Cell state update
            self.c = f_t * self.c + i_t * c_hat_t
            self.h = o_t * tf.tanh(self.c)
            
            outputs.append(self.h)

        output_sequence = tf.stack(outputs, axis=1)
        return self.out_dense(output_sequence) if self.return_sequences else output_sequence[:, -1, :]

    def __get_circuit(self) -> qml.QNode:
        def circuit(inputs: tf.Tensor, weights: tf.Tensor):
            # Improved feature embedding
            qml.AngleEmbedding(inputs, wires=range(self.wires), rotation='Y')
            qml.AngleEmbedding(inputs**2, wires=range(self.wires), rotation='Z')
            
            for layer in range(self.layers):
                # Simplified entangler pattern
                for i in range(self.wires-1):
                    qml.CNOT(wires=[i, i+1])
                qml.CNOT(wires=[self.wires-1, 0])
                
                # Parameterized rotations
                for i in range(self.wires):
                    qml.Rot(*weights[layer, 3*i:3*i+3], wires=i)
            
            return [qml.expval(qml.Z(i)) for i in range(self.wires)]

        return qml.QNode(circuit, dev.dev)
   
    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, input_shape[-2], self.units]) if self.return_sequences else tf.TensorShape([None, self.units])

    def build(self, input_shape):
        # Initialize Circuit embedding function
        self.__set_feature_function(self.feature_type)
        super().build(input_shape)
        
    def __set_feature_function(self, feature_type='atan'):
        if feature_type == 'atan':
            self.featue_function_1 = tf.atan
            self.featue_function_2 = lambda X: tf.atan(X**2)
        if feature_type == 'asin':
            self.featue_function_1 = tf.asin
            self.featue_function_2 = tf.acos
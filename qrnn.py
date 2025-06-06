from typing import Tuple
import pennylane as qml
import tensorflow as tf
import dev
import keras

dtype_global = tf.float64

class QRNN(keras.layers.Layer):

    def __init__(self, units, wires, layers, encoding='atan', return_sequences=True):
        super(QRNN, self).__init__()
        self.wires = wires
        self.units = units
        self.layers = layers
        self.encoding = encoding
        self.return_sequences = return_sequences
        self.featue_function_1, self.featue_function_2 = self.__get_encoding(encoding)

        self.circuit = self.__get_circuit()
        shapes = {'weights': (self.layers, 3*self.wires)}
        self.qlayers = qml.qnn.KerasLayer(self.circuit, shapes, output_dim=self.wires)
        
        self.feature_combiner = keras.layers.Dense(self.wires, dtype=dtype_global)
        self.out_dense = keras.layers.Dense(self.units, dtype=dtype_global)

    def __get_encoding(self, encoding='atan'):
        if encoding == 'atan':
            return tf.atan, lambda X: tf.atan(X**2)
        if encoding == 'asin':
            return tf.asin, tf.acos

    def __get_circuit(self):
        def circuit(inputs: Tuple[tf.Tensor, tf.Tensor], weights: tf.Tensor, ):
            inputs, state = inputs
            # Initialize State
            qml.StatePrep(state, wires=range(self.wires))
            
            # Angle embedding
            x_embedding_1 = self.featue_function_1(inputs)
            x_embedding_2 = self.featue_function_2(inputs)
            qml.AngleEmbedding(x_embedding_1, wires=range(self.wires), rotation="Y")
            qml.AngleEmbedding(x_embedding_2, wires=range(self.wires), rotation="Z")
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
            return [qml.expval(qml.Z(i)) for i in range(self.wires)], qml.state()

        return qml.QNode(circuit, dev.dev)

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        # initialize hidden state

        y_out = None
        state = None
        for i in range(inputs.shape[1]):
            # input -> (batch, time_steps, Features)
            y_t, state = self.rnn(inputs[:, :i+1], i)
            if y_out is None:
                y_out = y_t
            else:
                y_out = tf.concat([y_out, y_t], axis=1)
     
        # B*t*W -> (B, t, W)
        y_out = tf.reshape(y_out, ( inputs.shape[0], inputs.shape[1], self.wires ))
        y_out = self.out_dense(y_out)
        return y_out if self.return_sequences else y_out[:, -1, :], (self.c, self.h)

    def rnn(self, inputs: tf.Tensor, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Forward pass of the rnn, we save the output state of the last time step, which will be initialized as the next time step

        Args:
            inputs (tf.Tensor): input data
            state (tf.Tensor): previous state
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: output of the circuit, curent state
        """
        inputs = self.feature_combiner(inputs)
        
        return self.qlayers([inputs, state])
    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, input_shape[-2], self.units]) if self.return_sequences else tf.TensorShape([None, self.units])

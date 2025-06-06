from typing import Callable, Tuple
import tensorflow as tf

def create_dataset(
    transformation_function: Callable,
    sample_count:int = 100,
    timesteps: int = 4,
    start: float = 0.0,
    end: float = 2*tf.experimental.numpy.pi,
    dtype=tf.float64
    ) -> Tuple[tf.Tensor, tf.Tensor]:
    
    data = tf.linspace(start, end, sample_count + timesteps)
        
    data = transformation_function(data)
    X = []
    y = []
    for i in range(sample_count):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    X = tf.stack(X)
    y = tf.stack(y)
    X = tf.reshape(X, (sample_count, timesteps, 1))
    return X, y
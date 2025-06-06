import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.ops import summary_ops_v2

class GradTensorBoard(TensorBoard):
    def __init__(self, log_dir, loss=None, data=None, **kwargs):
        self.loss = loss
        self.data = data
        super().__init__(log_dir=log_dir, **kwargs)
    def on_batch_end(self, epoch, logs=None):
        model = self.model
        for x, y in self.data:
            continue
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(model.trainable_weights)
            z = model(x, training=True)
            loss = self.loss(y, z)
            grads = {w.name: tape.gradient(loss, w) for w in model.trainable_weights}
            with self._train_writer.as_default():
                for name, g in grads.items():
                    mean = tf.reduce_mean(tf.abs(g))
                    summary_ops_v2.scalar(f"epoch_grad_mean_{name}", mean, step=epoch)
                    tf.summary.histogram(f"epoch_grad_histogram_{name}", g, step=epoch)
        super().on_epoch_end(epoch, logs=logs)
        print("done")
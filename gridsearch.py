import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("qubits", type=int, default=0) # [10 12 14]
parser.add_argument("learning_rate", type=float, default=0.001) # [0.1 0.01 0.001]
parser.add_argument("version", type=str, default="") # [default, v2, reupload]
parser.add_argument("layers", default=2, type=int) # [2 3]
parser.add_argument("bidirectional", default='no', type=str) # [yes, no]

args = parser.parse_args()
print(args)

qubits = args.qubits
learning_rate = args.learning_rate
version = args.version
layers = args.layers
bidirectional = args.bidirectional



os.environ["OMP_NUM_THREADS"] = f"{qubits}"
os.environ["OMP_PROC_BIND"] = "false"
## Load other dependencies that might depend on config

import keras
import tensorflow as tf
import pennylane as qml
from q_lstm_tf import QLSTM as B_QLSTM
from q_lstm_tf_v2 import QLSTM as LE_QLSTM
from q_lstm_tf_v2_data_reuploaded import QLSTM as R_QLSTM
from q_lstm_tf_v2_data_reuploaded import BiQLSTM
from model_saver import DEFAUTL_MODEL_DICT, ModelSaver
from lbfgs_trainer import Lbfgs
from keras.layers import *
tf.keras.backend.set_floatx('float64')

seed = 42
tf.random.set_seed(seed)

path = f"qlstm.q{qubits}.lr{learning_rate}.v{version}.l{layers}.pkl"


file_path = Path(path)
model_weights = None
optim_state = None
model_dict = DEFAUTL_MODEL_DICT

if file_path.exists():
    import pickle
    printable = ['loss', 'accuracy', 'best_loss', 'best_score']
    with open(file_path, 'rb') as file:
        model_dict = pickle.load(file)
        model_weights = model_dict['current_model']
        optim_state = model_dict['optim_state']
        for key in printable:
            if key in model_dict:
                print(key, model_dict[key])


dev = qml.device('lightning.qubit', wires=qubits)

if bidirectional == 'yes':
    print("Using bidirectional witj reupload")
    qlstm = BiQLSTM(32, return_sequences=True, wires=qubits, layers=layers, dev=dev)
else:
    if version == "reupload":
        print(version)
        qlstm = R_QLSTM(32, return_sequences=True, wires=qubits, layers=layers, dev=dev)
    elif version == "v2":
        qlstm = LE_QLSTM(32, return_sequences=True, wires=qubits, layers=layers, dev=dev)
    else:
        qlstm = B_QLSTM(32, return_sequences=True, wires=qubits, layers=layers, dev=dev)

number_of_classes = 8
EPOCHS = 1
dtype_global = tf.float64


model = keras.Sequential([
    keras.layers.Input(shape=(55, 13), dtype=dtype_global),
    Conv1D(filters=32, kernel_size=3, activation="gelu", padding="same"),
    MaxPooling1D(pool_size=2, strides=2, padding="same"),
    Conv1D(filters=16, kernel_size=3, activation="gelu", padding="same"),
    MaxPooling1D(pool_size=2, strides=2, padding="same"),
    qlstm,
    TimeDistributed(Dense(16,activation='gelu')),
    TimeDistributed(Dense(8,activation='gelu')),
    Flatten(),
    Dense(number_of_classes,activation='gelu')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


print(model.summary())

model_saver = ModelSaver(path, model_dict)

train_spectrogram_ds = tf.data.experimental.load(
    'data/tf/train-2.ds', element_spec=None, compression=None, reader_func=None
)

val_spectrogram_ds = tf.data.experimental.load(
    'data/tf/val-2.ds', element_spec=None, compression=None, reader_func=None
)

test_spectrogram_ds = tf.data.experimental.load(
    'data/tf/test-2.ds', element_spec=None, compression=None, reader_func=None
)

batch_size = 64

train_spectrogram_ds.map(lambda x, y: (tf.cast(x, dtype=dtype_global), y))

val_spectrogram_ds.map(lambda x, y: (tf.cast(x, dtype=dtype_global), y))

test_spectrogram_ds.map(lambda x, y: (tf.cast(x, dtype=dtype_global), y))
if False:
    train_spectrogram_ds = train_spectrogram_ds.unbatch().filter(lambda x, y: y == 0 or y == 1).batch(batch_size).take(1)
    val_spectrogram_ds = val_spectrogram_ds.unbatch().filter(lambda x, y: y == 0 or y == 1).batch(batch_size).take(1)
    test_spectrogram_ds = test_spectrogram_ds.unbatch().filter(lambda x, y: y == 0 or y == 1).batch(batch_size).take(1)

train_spectrogram_ds = train_spectrogram_ds
val_spectrogram_ds = val_spectrogram_ds
test_spectrogram_ds = test_spectrogram_ds

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

# model.evaluate(test_spectrogram_ds.cache().take(1))

if optim_state is not None:
    print("Loading optim state ...")
    model.train_on_batch(tf.zeros((1, 55, 13)), tf.zeros((1, 1)))  # one dummy train step
    model.optimizer.set_weights(optim_state)
if model_weights is not None:
    print("Loading weights ...")
    ## hack for loading weights
    model.set_weights(model_weights)
    
# with open('test-' + path, 'wb') as file:
        
#     test_performance = model.evaluate(test_spectrogram_ds)
#     print("Test performance:", test_performance)
#     file.write(f"Test performance: {test_performance}\n")


history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(verbose=1, patience=2),
        model_saver
        ],
)

import pennylane as qml
import tensorflow as tf

qubits = 2
dev = qml.device('lightning.qubit', wires=qubits)
print(f"Using Quantum Device: {dev}")
from functools import partial
import math
from typing import Callable
import pennylane as qml
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from multiprocessing import Pool
from trainer import PyTorchTrainer
from torch.utils.data import Dataset, DataLoader
from util.copy_model import copy_model, load_model

dev = qml.device('lightning.qubit', wires=6 )
#torch.autograd.set_detect_anomaly(True)

#torch.random.manual_seed(42)
dtype_global = torch.float32


class QLstm(nn.Module):
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
        phi = []
        for _ in range(6): 
            
            weights = torch.normal(0, 1/torch.sqrt(torch.tensor(layers, dtype=dtype_global)), (layers, 3*wires), dtype=dtype_global)
            phi.append(weights)
        self.phi = torch.nn.Parameter(torch.stack(phi), requires_grad=True)
        
        self.feature_combiner = nn.Linear(feature_dim + wires, wires, dtype=dtype_global)
        self.sigmoid = nn.Sigmoid()
        
        self.__set_feature_function(type)
        
    
    def __set_feature_function(self, type='atan'):
        if type == 'atan':
            self.featue_function_1 = torch.atan
            self.featue_function_2 = lambda X: torch.atan(X**2)
        if type == 'asin':
            self.featue_function_1 = torch.asin
            self.featue_function_2 = torch.acos

    
    def layer(self, index, X: torch.Tensor):
        outputs = self.circuit(X, self.phi[index], self)
        return torch.stack(outputs).T
    
   
    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def circuit(X: torch.Tensor, weights: torch.Tensor, self):
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
    def rescale(self, X: torch.Tensor):
        return 2 * torch.sigmoid(X) - 1
    ### TODO: Batch input
    def lstm(self, X: torch.Tensor, c_t_1):
        """

        Args:
            X (torch.Tensor): shape of (B, F)
            c_t_1 (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Collapse hidden state and input state onto the dimensions of the wires
        # Make sure the output of the sigmoid is in (-1, 1)
        X = 2.0 * self.sigmoid(self.feature_combiner(X)) - 1.0 
        
        f_t = torch.sigmoid(self.layer(index=0, X=X))
        i_t = torch.sigmoid(self.layer(index=1, X=X))
        C_t = torch.sigmoid(self.layer(index=2, X=X))
        o_t = torch.sigmoid(self.layer(index=3, X=X)) 

        c_t = f_t * c_t_1 + i_t * C_t
        
        rescaled = o_t*torch.tanh(c_t)
        
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
    def forward(self, input: torch.Tensor):
        # initialize hidden state
        batch_size = input.shape[0]
        device = input.device
        self.h = torch.zeros((batch_size, self.wires), dtype=dtype_global).to(device)
        self.c =  torch.zeros((batch_size, self.wires), dtype=dtype_global).to(device)
        y_out = None
        for i in range(input.shape[1]): 
            # input -> (batch, time_steps, Features)
            y_t = self.lstm(torch.hstack([input[:, i], self.h]), self.c)
            if y_out == None:
                y_out = y_t
            else:
                y_out = torch.concat([y_out, y_t], axis=1)
                
        y_out = torch.reshape(y_out, ( input.shape[0], input.shape[1], self.wires )) # B*t*W -> (B, t, W)
        return y_out, (self.c, self.h)
    
class SimpleQLSTM(nn.Module):
    
    def __init__(self, wire_count: int = 6, layer_count: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(wire_count, 1, dtype=dtype_global)
        self.qlstm = QLstm(wire_count , layer_count, type='atan')
        
    def forward(self, input: torch.Tensor):

        input = input.unsqueeze(-1)  # Add feature dimension

        Y , _ = self.qlstm(input)
        Y = self.linear(Y[:, -1])
        
        return Y
    
class SimpleLSTM(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.linear = nn.Linear(500, 1).to(dtype=dtype_global)
        self.lstm = nn.LSTM(1 , 500, 1, batch_first=True).to(dtype=dtype_global)
        
    def forward(self, input: torch.Tensor):
        input = input.unsqueeze(-1)  # Add feature dimension
        output, (hn, cn) = self.lstm(input)
        output = self.linear(output[:, -1])
        return output

class TestDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

if __name__ == '__main__':
    
    
    data = torch.linspace(0, 4*torch.pi, 54, dtype=dtype_global)
    
    data = torch.sin(data)
    dataset = []
    expected = []
    for i in range(data[:-4].shape[0]):
        dataset.append(data[i:i+4])
        expected.append(data[i+4])
    dataset = torch.stack(dataset)
    expected = torch.stack(expected)
    
    plt.plot(expected)
    plt.show()
    
    loss = 0.0
    
    #out = qlstm.call(dataset[0])
    #print(f"Test pass through: {out}")
    # for i in range(10):
    #     for j in range(expected.shape[0]):
    #         y_pred = qlstm.call(dataset[j])
    #         loss += loss(y_pred, expected[j])    

    #     optimizer.step()
    simpleLSTM = SimpleQLSTM(3, 3)
    
    loss_fn = torch.nn.MSELoss()
    
    for arr in list(simpleLSTM.parameters()):
        print(arr.size())
    
    optim = torch.optim.Adam(simpleLSTM.parameters(), lr=0.1)
    
    loss_history = []
    
    model_best = None
    
    

    qlstm_trainer = PyTorchTrainer(simpleLSTM, DataLoader(TestDataset(dataset, expected)), loss_fn, optim, epochs=100)
    qlstm_trainer.fit()
    # t = trange(100, desc='Loss: ', leave=True)
    # simpleLSTM.train()
    # for _ in t:
    #     def closure():
            

    #         optim.zero_grad()
            
            
                
    #         out = simpleLSTM(dataset)
    #         loss = loss_fn(out.squeeze(), expected)
            
    #         #l2_norm = sum(p.pow(2.0).mean() for p in simpleLSTM.parameters())    
    #         #loss += l2_norm * 0.00001
            
            
    #         loss_history.append(loss.item())
    #         t.set_description(f"Loss: {loss.item():.4f}")

    #         loss.backward()
    #         return loss
    #     optim.step(closure)


    

    simpleLSTM.eval()

    data = torch.linspace(10*torch.pi, 15*torch.pi, 100, dtype=dtype_global)

    data += torch.normal(0, 0.1, size=data.shape)
    data = 0.25 * (torch.sin(data) + torch.cos(1.5*data)) 
    dataset = []
    expected = []
    for i in range(data[:-4].shape[0]):
        dataset.append(data[i:i+4])
        expected.append(data[i+4])
    dataset = torch.stack(dataset)
    expected = torch.stack(expected)
    
    y_pred = []
    out = simpleLSTM(dataset)
    y_pred = np.array(out.detach().numpy())
    
    plt.title("Loss history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.show()
    
    plt.title("Test dataset: sinusoid")
    plt.plot(expected)
    plt.plot(y_pred, linestyle='dashed')
    plt.legend(['expected', 'predicted'])
    plt.show()
    

    
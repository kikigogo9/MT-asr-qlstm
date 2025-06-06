from copy import deepcopy
import torch

def copy_model(model, loss, model_name='qlstm'):
    path = f"./{model_name}/loss-{loss}.torch"
    torch.save(model.state_dict(), path)
    
    return (path, loss)
    
def load_model(path, model: torch.nn.Module):
    model.load_state_dict( path, weights_only=True)
    return model
import tensorflow as tf
from keras.callbacks import Callback
import pickle


get_model_dict = lambda: DEFAUTL_MODEL_DICT.copy()

DEFAUTL_MODEL_DICT = {
    'loss': [],
    'accuracy': [],
    'best_model': None,
    'best_score': None,
    'best_loss': None,
    'optim_state': None,
    'current_model': None,
}

class ModelSaver(Callback):
    file_path: str
     
    def __init__(self, file_path: str, model_dict: dict = None) -> None:
        super().__init__()
        self.file_path = file_path
        if model_dict is None:
            self.model_dict = DEFAUTL_MODEL_DICT.copy()
        else:
            self.model_dict = model_dict
    
    def on_epoch_end(self, epoch, logs=None):     
        pickle.dump(self.__update_model_dict(logs), open(self.file_path, 'wb'))
        
        return super().on_epoch_end(epoch, logs)
    
    def __update_model_dict(self, logs) -> dict:
        if logs is None:
            return self.model_dict
        
        self.model_dict['loss'].append(logs['val_loss'])
        if 'val_accuracy' in logs:
            self.model_dict['accuracy'].append(logs['val_accuracy'])
        
        if self.model_dict['best_loss'] is None or self.model_dict['best_loss'] > logs['val_loss']:
            self.model_dict['best_loss'] = logs['val_loss']
            if 'val_accuracy' in logs:
                self.model_dict['best_score'] = logs['val_accuracy']
            self.model_dict['best_model'] = self.model.get_weights()
            self.model_dict['optim_state'] = self.model.optimizer.variables()

        self.model_dict['current_model'] = self.model.get_weights()

        return self.model_dict
    
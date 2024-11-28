
class Trainer:
    '''
    This class is used to define the interface for training the model
    '''
    def __init__(self):
        pass

    def train(self, num_iterations: int):
        '''
        This method is used to train the model
        '''
        raise NotImplementedError

    def load_model(self, model_path: str):
        '''
        Load the model
        @param model_path: Path to the model
        @return: None
        '''
        raise NotImplementedError

    def save_model(self):
        '''
        Save the model
        '''
        raise NotImplementedError
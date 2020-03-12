import numpy as np
import matplotlib.pyplot as plt

class Data():
    def __init__(self, input_path, n_test):
        
        file_content = np.genfromtxt(input_path, delimiter=' ', dtype=np.float32)
        self.x_test = file_content[:n_test,:-1].reshape(-1,1,14,14)/255
        self.x_train = file_content[n_test:,:-1].reshape(-1,1,14,14)/255
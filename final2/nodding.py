import numpy as np
#changed threshold from 75 to 10

def isNodding(vector_buffer):
    vector_buffer = np.asarray(vector_buffer)
    vector_buffer = vector_buffer[:, 1]
    variance = np.var(vector_buffer)
    return variance > 10

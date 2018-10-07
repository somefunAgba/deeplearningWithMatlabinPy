import time

import numpy as np

from deeplearning_improves.number_recogition_deepnn_trainer import back_prop_ce_multi_class as bpnn

np.set_printoptions(formatter={'float': '{: 0.1f}'.format}, precision=4)

# USER TESTING ... INFERENCE
# Test Data
N1 = np.array([
    [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1]
])
N2 = np.array([
    [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]
])
N3 = np.array([
    [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1]
])
N4 = np.array([
    [0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 1, 0]
])
N5 = np.array([
    [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1]
])
I1 = np.array([
    [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1]
])
I2 = np.array([
    [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]
])
I3 = np.array([
    [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0]
])
I4 = np.array([
    [0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 1, 0]
])
I5 = np.array([
    [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0]
])
C1 = np.array([
    [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0]
])
C2 = np.array([
    [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]
])
C3 = np.array([
    [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0]
])
C50 = np.array([
    [0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0]
])
C5 = np.array([
    [0, 1, 1, 1, 1], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 0]
])
X = np.array([N1, N2, N3, N4, N5, I1, I2, I3, I4, I5, C1, C2, C3, C50, C5])
X = X.reshape([len(X), int(np.size(X) / len(X))])

# Neural Network Inference
y = 0
try:
    bp = bpnn
    ti = time.clock()
    y = bp('test', X)
    tf = time.clock()
    print(f'Time Used: {(tf-ti):.2e} secs', '\n')

    # Mapping of binary output values to its real-world meaning or value
    yval = np.max(y, axis=1)
    yval_index = np.argmax(y, axis=1)
    recognized_number = np.zeros(len(X))

    i = 0
    for index in yval_index:
        if index == 0:
            # recognized_number[i] = index + 1
            recognized_number.put([i], index + 1)
        elif index == 1:
            recognized_number.put([i], index + 1)
        elif index == 2:
            recognized_number.put([i], index + 1)
        elif index == 3:
            recognized_number.put([i], index + 1)
        elif index == 4:
            recognized_number.put([i], index + 1)
        else:
            recognized_number[i] = -1
            print('Number may not be within the range of 0 - 5')
        i = i + 1

    if len(X) > 1:
        print('NUMBER RECOGNITION OF 5 * 5 BLACK AND WHITE IMAGES: ')
    else:
        print('NUMBER RECOGNITION OF 5 * 5 BLACK AND WHITE IMAGE:')

    # def formatter(p):
    #     return "%.1f" % p

    # Visualize Results
    ns = int(np.sqrt(len(X[0])))
    div = '-' * 60
    print('IMAGE'.center(15), 'RECOGNITION as NUMBER'.center(50), '\n', div, end='\n')
    for _x, _y, _z in zip(X, recognized_number, yval * 100):
        print(_x.reshape([ns, ns]), ':'.ljust(5),
              'BW Image => ', _y, '|'.center(5),
              'Inferred Probability(%) :'.rjust(20),
              '{0:.1f}'.format(_z), '\n', div, end='\n')

except:
    print('Must be that the Neural Network has not been trained yet!')

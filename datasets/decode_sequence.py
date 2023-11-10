import numpy as np

filename = "2kHz_32.npy"

sequence = np.load(filename)

print(filename)
print(len(sequence))

valid_start_index = np.where(sequence != -1)[0][0]
print(valid_start_index)

start_code_detection = []

start_code = str(np.array([1,1,1,0]))

for idx in range(len(sequence)) : 
    if idx < valid_start_index :
        start_code_detection.append(-1)
    elif str(sequence[idx-4:idx]) == start_code :    
        start_code_detection.append(1)
        print(idx)
    else :
        start_code_detection.append(0)

print(len(start_code_detection))
print(np.where(start_code_detection[start_code_detection==0]))

print(start_code_detection[:100])


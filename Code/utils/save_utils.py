import pickle
from tensorflow.keras.optimizers import  Optimizer


# -------------------------------------------------- READ/WRITE PKL FILES ------------------------------------------------------------
# store processed data in pkl files
def save_pkl_data(data, filename, protocol=pickle.HIGHEST_PROTOCOL):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol)
        print("data stored succesfully to: ", filename)


# read processed data in pkl files
def load_pkl_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def load_parameters(filename):
    params = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            line_values = line.split('=')
            if len(line_values) < 2:
                raise ValueError('Wrong Parameter assign value. Expected formar <name> = <value>')
            params[line_values[0]] = int(line_values[1])

    return params

import pickle


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


import pickle
import os


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


def valid_paths(*paths):
    for path in paths:
        if not os.path.isfile(path):
            raise RuntimeError('[ERR]: ' + path + ' is not valid file')
    return True


def load_parameters(filename):
    params = {}
    with open(filename, 'r') as file:
        # traverse each line
        for i, line in enumerate(file.readlines()):
            i += 1
            line = line.strip()
            # empty line or comment line
            if len(line) == 0 or line[0] == '#':
                continue
            input_line = line.split('=')
            # verify = token in line
            if len(input_line) < 2:
                raise RuntimeError('[ERR] Wrong Parameter Line: ' + str(i) + '. Expected format <name> : <type> = <value>. '
                                   'Missing ":". File: ' + filename)
            # verify : token in line
            id_type = input_line[0].split(':')
            if len(id_type) < 2:
                raise RuntimeError('[ERR] Wrong Parameter Line: ' + str(i) + '. Expected format <name> : <type> = <value>. '
                                   'Missing ":". File: ' + filename)
            # get id, type and value
            id_, type_ = id_type
            value = input_line[1]
            # remove spaces and new lines (\n)
            id_ = id_.strip()
            type_ = type_.strip()
            value = value.strip()
            # verify correct type
            if type_ not in ('int', 'float', 'str', 'bool'):
                raise RuntimeError('[ERR] Wrong Parameter Line:' + str(i) + ' type: ' + type_ + '. Types supported are '
                                   '"int", "float", "str", "bool". File: ' + filename)
            # CAST TO DESIRED TYPE
            if type_ == 'int':
                value = int(value)
            elif type_ == 'float':
                value = float(value)
            elif type_ == 'bool':
                value = bool(value)
            params[id_] = value

    return params

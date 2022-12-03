import numpy as np

def load_data(file):
    data = {}
    cont = 0
    for row in file.readlines():
        item = row.split()
        if not item: continue
        t = int(float(item[0])) #es el tiempo actual
        idx = int(float(item[1])) #Es el indice -1 para la pelota, y el resto el numero de jugador
        x = float(item[2]) #posicion x del objeto
        y = float(item[3]) #posicion y del objeto
        if t not in data: #Si no se ha agregado este tiempo, lo creamos
            data[t] = {}
        #Como los primers 5 jugadores son de un equipo
        if cont < 5:
            team = 0
        else:
            team = 1
        #Si estamos observando la pelota, se reinicia el contador y no se considera de ningun equipo
                
        if idx == -1:
            team = -1
            cont = 0
        else:
            cont = cont+1

        data[t][idx] = [x, y, team]
    #Nos devuelve arreglos en donde dado el tiempo y el agente, nos da su posicion
    return data

#La sigueinte funcion transforma los datos de tal manera que la red puede leerlos, 
#lo que hace realmeente es reindexar
#Si le mandamos un 1 agrega una columna que diferencia los equipos y el balon
def to_cube(file, team = 0):
    data = load_data(file)
    d_1 = len(data)
    if(team):
        A = np.zeros((d_1, 11, 3))
    else:
        A = np.zeros((d_1, 11, 2))
    c_time = 0
    c_player = 0
    for t in data:
        for p in data[t]:
            A[c_time][c_player][0] = data[t][p][0]
            A[c_time][c_player][1] = data[t][p][1]
            if team:
                A[c_time][c_player][2] = data[t][p][2]
            c_player += 1
        c_time += 1
        c_player = 0
    return A

#le pasamos cuanto tiempo queremos que vea del pasado y cuanto del futuro
#Separamos en dos bloques del mismo tama;o, y los datos que no tomaremos en cuenta se llenan de 1s
#es decir los ultimos del bloque con timepo menor
def separete_data(cube, past_time, future_time):
    time = max(past_time, future_time)
    n_players = len(cube[0])
    s = len(cube[0][0])
    cube_past = np.zeros((time, n_players, s))
    cube_future = np.zeros((time, n_players, s))
    aux = np.split(cube, [past_time, past_time + future_time])
    cube_future[0:future_time]  = aux[1]
    cube_past[0:past_time] = aux[0]

    print(np.shape(cube_past))
    print(np.shape(cube_future))

    return cube_past, cube_future

def mask(past_time, future_time):
    time = max(past_time, future_time)
    past = np.ones(time)
    future = np.ones(time)
    #1 indica a la maquina que no lea esos valores
    past[0:past_time] = 0
    future[0:future_time] = 0
    return past, future

if __name__ == "__main__":
    import os
    current_directory = os.getcwd()
    file = open("Code/dataset/NBA_data/ATL_BOS.txt", "r")
    #d = load_data(f)
    #for idx in d[0]:
    #    print(idx, ": ", d[0][idx])
    cube = to_cube(file)
    #print(cube)
    p, f = mask(10, 22-10)
    print("past:")
    print(p)
    print("future")
    print(f)

    
    
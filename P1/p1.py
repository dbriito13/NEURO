import random 
import sys

class Neurona:
    def __init__(self, umbral, tipo, nombre, entrenando=True):
        self.umbral = umbral
        self.tipo = tipo
        self.valor_entrada = 0
        self.valor_salida = 0
        self.conexiones = []
        self.nombre = nombre
        self.entrando = entrenando

    def inicializar(self, x):
        self.valor_entrada = x
        
    def conectar(self, neurona, peso):
        conexion = Conexion(peso, neurona)
        self.conexiones.append(conexion)
        
    def disparar(self):
        if self.tipo == "Entrada" or (self.tipo=="Adaline" and self.entrenando==True):
            self.valor_salida = self.valor_entrada
        elif self.tipo == "Perceptron":
            if self.valor_entrada > self.umbral:
                self.valor_salida = 1
            elif self.valor_entrada >= -self.umbral:
                self.valor_salida = 0
            else:
                self.valor_salida = -1
        elif self.tipo == "McCulloch-Pitts":
            if self.valor_entrada >= self.umbral:
                self.valor_salida = 1
            else:
                self.valor_salida = 0
        elif self.tipo == "Adaline":
            if self.valor_entrada >= 0:
                self.valor_salida = 1
            else:
                self.valor_salida = -1
    def propagar(self):
        for conexion in self.conexiones:
            conexion.propagar(self.valor_salida)
        

class Conexion:
    def __init__(self, peso, neurona):
        self.peso = peso
        self.peso_anterior = 0
        self.valor = 0
        self.neurona = neurona
    
    def propagar(self, valor):
        self.neurona.valor_entrada += valor * self.peso



class Capa:
    def __init__(self):
        self.neuronas = []
    
    def inicializar(self):
        for neurona in self.neuronas:
            neurona.inicializar(0)

    def anyadir(self, neurona):
        self.neuronas.append(neurona)
    
    def anyadir_lista(self, neuronas):
        for neurona in neuronas:
            self.neuronas.append(neurona)

    def conectar_capa(self, capa, peso_min, peso_max):
        for neurona_destino in capa.neuronas:
            self.conectar_neurona(neurona_destino, peso_min, peso_max)

    def conectar_neurona(self, neurona, peso_min, peso_max):
        # Revisar
        # Mirar libreria random
        peso = random.random() 
        peso = peso*(peso_max-peso_min) + peso_min

        for neurona_origen in self.neuronas:
                neurona_origen.conectar(neurona, peso)

    def disparar(self):
        for neurona in self.neuronas:
            neurona.disparar()
    
    def propagar(self):
        for neurona in self.neuronas:
            neurona.propagar()
        

class RedNeuronal:
    def __init__(self):
        self.capas = []
    
    def anyadir(self, capa):
        self.capas.append(capa)
    
    def inicializar(self):
        for capa in self.capas:
            capa.inicializar()

    #En el propagar en redneuronal llamar a inicializar(0) en todas las neuronas de la siguiente capa
    def disparar(self):
        for capa in self.capas:
            capa.disparar()

    def propagar(self):
        for capa in self.capas:
            capa.inicializar()
        
        for capa in self.capas:
            capa.propagar()

    def mostrar_nombres(self, fichero):
        tipo = self.capas[-1].neuronas[0].tipo
        if tipo == "McCulloch-Pitts":
            for capa in self.capas:
                for neurona in capa.neuronas:
                    fichero.write(neurona.nombre + " ")
            fichero.write("\n")
        elif tipo == "Perceptron":
            for capa in self.capas:
                for neurona in capa.neuronas:
                    fichero.write(neurona.nombre + " ")
            
            n_entradas = len(self.capas[0].neuronas)
            n_salidas = len(self.capas[1].neuronas)
            if n_salidas == 1:
                for i in range(n_entradas):
                    if i != n_entradas-1:
                        fichero.write("w_"+ str(i+1) + " ")
                    else:
                        fichero.write("b ")
            else:
                for i in range(n_salidas):
                    for j in range(n_entradas):
                        if j != n_entradas-1:
                            fichero.write("w_"+ str(i+1)+ "_" + str(j+1) + " ")
                        else:
                            fichero.write("b_"+ str(i+1) + " ")
            fichero.write("\n")

    def mostrar_estado(self, fichero):
        tipo = self.capas[-1].neuronas[0].tipo
        if tipo == "McCulloch-Pitts":
            for capa in self.capas:
                for neurona in capa.neuronas:
                    fichero.write(str(neurona.valor_salida) + " ")
            fichero.write("\n")
        elif tipo == "Perceptron":
            for neurona in self.capas[0].neuronas:
                fichero.write(str(neurona.valor_salida) + " ")
            for neurona in self.capas[1].neuronas:
                fichero.write(str(neurona.valor_salida) + " ")
            for neurona in self.capas[0].neuronas:
                fichero.write(str(neurona.conexiones[0].peso) + " ")
            fichero.write("\n")


def leer2(fichero):
    first_line = fichero.readline()
    n_entrada = int(first_line[0])
    n_salida = int(first_line[2])
    entradas_datos = []
    salidas_datos = []
    for ln in fichero:
        linea = ln.split(' ')
        linea[-1] = linea[-1][:-1]
        linea = list(map(float, linea))
        entradas_datos.append(linea[:n_entrada])
        salidas_datos.append(linea[n_entrada:])
    return (entradas_datos, salidas_datos)

def leer1(fichero, por):
    first_line = fichero.readline()
    n_entrada = int(first_line[0])
    n_salida = int(first_line[2])
    lineas = []
    for linea in fichero:
        lineas.append(linea)

    random.shuffle(lineas)

    entradas_entrenamiento = []
    salidas_entrenamiento = []
    entradas_test = []
    salidas_test = []

    num_lineas_entrenamiento = int(len(lineas) * por)
    for ln in lineas[:num_lineas_entrenamiento]:
        linea = ln.split(' ')
        linea[-1] = linea[-1][:-1]
        linea = list(map(float, linea))
        entradas_entrenamiento.append(linea[:n_entrada])
        salidas_entrenamiento.append(linea[n_entrada:])
    
    for ln in lineas[num_lineas_entrenamiento:]:
        linea = ln.split(' ')
        linea[-1] = linea[-1][:-1]
        linea = list(map(float, linea))
        entradas_test.append(linea[:n_entrada])
        salidas_test.append(linea[n_entrada:])

    return entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test

def leer3(fichero_entrenamiento, fichero_test):
    entradas_entrenamiento, salidas_entrenamiento = leer2(fichero_entrenamiento)
    entradas_test, salidas_test = leer2(fichero_test)
    return entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test
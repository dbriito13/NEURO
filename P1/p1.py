import random 
import sys

class Neurona:
    def __init__(self, umbral, tipo, nombre):
        self.umbral = umbral
        self.tipo = tipo
        self.valor_entrada = 0
        self.valor_salida = 0
        self.conexiones = []
        self.nombre = nombre

    def inicializar(self, x):
        self.valor_entrada = x
        
    def conectar(self, neurona, peso):
        conexion = Conexion(peso, neurona)
        self.conexiones.append(conexion)
        
    def disparar(self):
        if self.tipo == "Perceptron":
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
        for capa in self.capas:
            for neurona in capa.neuronas:
                fichero.write(neurona.nombre + "\t")
        fichero.write("\n")

    def mostrar_estado(self, fichero):
        tipo = self.capas[0].neuronas[0].tipo
        if tipo == "McCulloch-Pitts":
            for capa in self.capas:
                for neurona in capa.neuronas:
                    fichero.write(str(neurona.valor_salida) + "\t")
            fichero.write("\n")
        elif tipo == "Perceptron":
            for neurona in self.capas[0].neuronas:
                fichero.write(str(neurona.valor_salida) + "\t")
            for neurona in self.capas[1].neuronas:
                fichero.write(str(neurona.valor_salida) + "\t")
            for neurona in self.capas[0].neuronas:
                fichero.write(str(neurona.conexiones[0].peso) + "\t")
            fichero.write("\n")


def leer2(fichero):
    first_line = fichero.readline()
    n_entrada = int(first_line[0])
    n_salida = int(first_line[2])
    entradas_datos = []
    salidas_datos = []
    for ln in fichero:
        linea = ln.split('  ')
        linea[-1] = linea[-1][:-1]
        linea = list(map(float, linea))
        entradas_datos.append(linea[:n_entrada])
        salidas_datos.append(linea[n_entrada:])
    return (entradas_datos, salidas_datos)
from p1 import *
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error, numero de argumentos incorrecto!")
        exit(1)

    #Creamos una instancia de la red neuronal que vamos a implementar
    red_neuronal = RedNeuronal()
    capa = Capa()

    #Neuronas
    x1 = Neurona(1, "McCulloch-Pitts", "x1")
    x2 = Neurona(1, "McCulloch-Pitts", "x2")

    z1 = Neurona(2, "McCulloch-Pitts", "z1")
    z2 = Neurona(2, "McCulloch-Pitts", "z2")

    y1 = Neurona(2, "McCulloch-Pitts", "y1")
    y2 = Neurona(2, "McCulloch-Pitts", "y2")
    capa.anyadir_lista([x1, x2, z1, z2, y1, y2])

    #Conexiones
    x2.conectar(z1, -1)
    x2.conectar(z2, 2)
    x2.conectar(y2, 1)

    x1.conectar(y1, 2)

    z1.conectar(y1, 2)

    z2.conectar(y2, 1)
    z2.conectar(z1, 2)

    red_neuronal.anyadir(capa)
    red_neuronal.inicializar()

    #Leemos archivo de entrada
    f_entrada = open(sys.argv[1], "r")
    f_salida = open(sys.argv[2], "w")
    red_neuronal.mostrar_nombres(f_salida)
    for ln in f_entrada:
        x1.inicializar(int(ln[0]))
        x2.inicializar(int(ln[2]))
        red_neuronal.disparar()
        red_neuronal.propagar()
        red_neuronal.mostrar_estado(f_salida)

    f_entrada.close()
    f_salida.close()
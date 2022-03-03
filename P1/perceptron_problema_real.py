from cProfile import label
from p1 import *
import sys
from matplotlib import pyplot as plt


def calcular_predicciones(red_neuronal, entradas):
    n_entradas = len(entradas[0])
    predicciones = []
    for i in range(0, len(entradas)):
        for j in range(n_entradas):
            entrada = entradas[i][j]
            neurona = red_neuronal.capas[0].neuronas[j]
            neurona.inicializar(entrada)
        red_neuronal.disparar()
        red_neuronal.propagar()
        red_neuronal.capas[1].disparar()
        prediccion = []
        for neurona in red_neuronal.capas[1].neuronas:
            prediccion.append(neurona.valor_salida)
        predicciones.append(prediccion)
    return predicciones


def error_cuadratico(red_neuronal, entradas, salidas):
    predicciones = calcular_predicciones(red_neuronal, entradas)
    num_predicciones = len(predicciones)
    tamano_predicciones = len(predicciones[0])
    err_total = 0
    for i in range(num_predicciones):
        err = 0
        real = salidas[i]
        predicho = predicciones[i]
        for j in range(len(real)):
            err += (real[j] - predicho[j])**2
        err_total += err
    err_total /= (num_predicciones * tamano_predicciones)
    return err_total


def main():
    if len(sys.argv) != 5:
        print("Error, número de argumentos incorrecto!")
        exit(1)

    num_problema = int(sys.argv[1])
    max_epochs = int(sys.argv[2])
    umbral = float(sys.argv[3])
    tasa = float(sys.argv[4])
    
    if num_problema == 1:
        fichero = open("problema_real1.txt", "r")
        entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test = leer1(fichero, 0.7)
    elif num_problema == 2:
        fichero_entrenamiento = open("problema_real2.txt", "r")
        fichero_test = open("problema_real2_no_etiquetados.txt", "r")
        entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test = leer3(fichero_entrenamiento, fichero_test)

    for i in range(len(entradas_entrenamiento)):
        entradas_entrenamiento[i].append(1)
    for i in range(len(entradas_test)):
        entradas_test[i].append(1)

    n_entradas = len(entradas_entrenamiento[0])
    n_salidas = len(salidas_entrenamiento[0])

    red_neuronal = RedNeuronal()
    capa_entrada = Capa()
    capa_salida = Capa()

    for i in range(n_entradas):
        if i != n_entradas-1:
            capa_entrada.anyadir(Neurona(umbral, "Entrada", "x" + str(i+1)))
        else:
            capa_entrada.anyadir(Neurona(umbral, "Entrada", "1"))

    for i in range(n_salidas):
        capa_salida.anyadir(Neurona(umbral, "Perceptron", "y" + str(i+1)))

    capa_entrada.conectar_capa(capa_salida, 0, 0)
    red_neuronal.anyadir(capa_entrada)
    red_neuronal.anyadir(capa_salida)
    red_neuronal.inicializar()

    max_incremento_pesos = 1
    n_epoch = 0
    error_entrenamiento = []
    error_test = []
    while max_incremento_pesos > 0 and n_epoch < max_epochs:
        max_incremento_pesos = 0
        for i in range(0, len(entradas_entrenamiento)):
            for j in range(n_entradas):
                entrada = entradas_entrenamiento[i][j]
                neurona = capa_entrada.neuronas[j]
                neurona.inicializar(entrada)
            red_neuronal.disparar()
            red_neuronal.propagar()
            capa_salida.disparar()

            for j in range(n_salidas):
                t = salidas_entrenamiento[i][j]
                y = capa_salida.neuronas[j].valor_salida
                if y != t:
                    for neurona in capa_entrada.neuronas:
                        max_incremento_pesos = max(max_incremento_pesos, abs(tasa*t*neurona.valor_salida))
                        neurona.conexiones[j].peso += tasa*t*neurona.valor_salida
        n_epoch += 1

        err = error_cuadratico(red_neuronal, entradas_entrenamiento, salidas_entrenamiento)
        error_entrenamiento.append(err)
        if num_problema == 1:
            err = error_cuadratico(red_neuronal, entradas_test, salidas_test)
            error_test.append(err)

    if num_problema == 1:
        plt.plot(range(len(error_entrenamiento)), error_entrenamiento, label="Entrenamiento")
        plt.plot(range(len(error_test)), error_test, label="Test")
        plt.legend()
        plt.show()
    elif num_problema==2:
        plt.plot(range(len(error_entrenamiento)), error_entrenamiento)
        plt.title("Entrenamiento")
        plt.show()

        fichero_predicciones = open("prediccion_perceptron.txt", "w")
        predicciones = calcular_predicciones(red_neuronal, entradas_test)
        for prediccion in predicciones:
            fichero_predicciones.write(str(prediccion[0]) + " " + str(prediccion[1]) + "\n")


if __name__ == "__main__":
    main()
    
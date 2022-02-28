from p1 import *
import sys


def error_cuadratico(red_neuronal, entradas_test, salidas_test):
    fichero = open("temp.txt", "w")
    n_entradas = len(entradas_test[0])
    n_salidas = len(salidas_test[0])
    predicciones = []
    for i in range(0, len(entradas_test)):
        print(entradas_test[i])
        for j in range(n_entradas):
            entrada = entradas_test[i][j]
            neurona = red_neuronal.capas[0].neuronas[j]
            neurona.inicializar(entrada)
        red_neuronal.disparar()
        red_neuronal.propagar()
        capa_salida.disparar()

        prediccion = []
        for neurona in red_neuronal.capas[1].neuronas:
            prediccion.append(neurona.valor_salida)
        predicciones.append(prediccion)
        red_neuronal.mostrar_estado(fichero)


    num_datos = len(salidas_test)
    err_total = 0
    for i in range(num_datos):
        err = 0
        real = salidas_test[i]
        predicho = predicciones[i]
        for j in range(len(real)):
            err += (real[j] - predicho[j])**2
        err_total += err
    err_total /= num_datos
    return err_total


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Error, número de argumentos incorrecto!")
        exit(1)

    fichero = "problema_real1.txt"
    max_epochs = int(sys.argv[1])
    umbral = float(sys.argv[2])
    tasa = float(sys.argv[3])
    
    fichero = open(fichero, "r")
    entradas_datos, salidas_datos, entradas_test, salidas_test = leer1(fichero, 0.7)
    for i in range(len(entradas_datos)):
        entradas_datos[i].append(1)

    n_entradas = len(entradas_datos[0])
    n_salidas = len(salidas_datos[0])

    red_neuronal = RedNeuronal()
    capa_entrada = Capa()
    capa_salida = Capa()

    for i in range(n_entradas):
        if i != n_entradas-1:
            capa_entrada.anyadir(Neurona(umbral, "Perceptron", "x" + str(i+1)))
        else:
            capa_entrada.anyadir(Neurona(umbral, "Perceptron", "1"))

    for i in range(n_salidas):
        capa_salida.anyadir(Neurona(umbral, "Perceptron", "y" + str(i+1)))

    capa_entrada.conectar_capa(capa_salida, 0, 0)
    red_neuronal.anyadir(capa_entrada)
    red_neuronal.anyadir(capa_salida)
    red_neuronal.inicializar()

    #capa_entrada.anyadir_lista([Neurona(umbral, "Perceptron", "x" + str(i+1)) for i in range(n_entradas)])
    f_salida = open("salida_" + sys.argv[1], "w")
    max_incremento_pesos = 1
    n_epoch = 0
    red_neuronal.mostrar_nombres(f_salida)
    while max_incremento_pesos > 0 and n_epoch < max_epochs:
        max_incremento_pesos = 0
        for i in range(0, len(entradas_datos)):
            for j in range(n_entradas):
                entrada = entradas_datos[i][j]
                neurona = capa_entrada.neuronas[j]
                neurona.inicializar(entrada)
            red_neuronal.disparar()
            red_neuronal.propagar()
            capa_salida.disparar()

            for j in range(n_salidas):
                t = salidas_datos[i][j]
                y = capa_salida.neuronas[j].valor_salida
                if y != t:
                    for neurona in capa_entrada.neuronas:
                        max_incremento_pesos = max(max_incremento_pesos, abs(tasa*t*neurona.valor_salida))
                        neurona.conexiones[j].peso += tasa*t*neurona.valor_salida
            red_neuronal.mostrar_estado(f_salida)
        n_epoch += 1
        err = error_cuadratico(red_neuronal, entradas_test, salidas_test)
        print(err)



    
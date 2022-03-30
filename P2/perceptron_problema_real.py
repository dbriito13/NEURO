from cgi import test
from p2 import *
import sys
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sn


def normalizar(entradas_entrenamiento, entradas_test):
    df_entrenamiento = pd.DataFrame(entradas_entrenamiento)
    mu = df_entrenamiento.mean()
    sigma = df_entrenamiento.std()
    df_entrenamiento = (df_entrenamiento - mu)/sigma
    df_test = pd.DataFrame(entradas_test)
    df_test = (df_test - mu)/sigma
    return df_entrenamiento.values.tolist(), df_test.values.tolist()


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
        red_neuronal.disparar()
        red_neuronal.propagar()
        red_neuronal.capas[-1].disparar()
        prediccion = []
        for neurona in red_neuronal.capas[-1].neuronas:
            prediccion.append(neurona.valor_salida)
        predicciones.append(prediccion)
    return predicciones


def error_cuadratico(predicciones, salidas):
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


def tasa_aciertos(predicciones, salidas):
    num_predicciones = len(predicciones)
    aciertos = 0
    for i in range(num_predicciones):
        real = salidas[i]
        predicho = predicciones[i]
        clase_real = real.index(max(real))
        clase_predicha = predicho.index(max(predicho))
        if clase_real == clase_predicha:
            aciertos += 1
    return aciertos / num_predicciones


def calcular_matriz_confusion(predicciones, salidas):
    num_clases = len(predicciones[0])
    num_predicciones = len(predicciones)
    m = np.zeros((num_clases, num_clases))
    for i in range(num_predicciones):
        real = salidas[i]
        predicho = predicciones[i]
        clase_real = real.index(max(real))
        clase_predicha = predicho.index(max(predicho))
        m[clase_predicha, clase_real] += 1
    return m/num_predicciones


def main():
    if len(sys.argv) != 5:
        print("Error, número de argumentos incorrecto!")
        exit(1)

    num_problema = int(sys.argv[1])
    max_epochs = int(sys.argv[2])
    tasa = float(sys.argv[3])
    p = int(sys.argv[4])
    
    if num_problema == 0:
        fichero_entrenamiento = open("entrada/problema_real2.txt", "r")
        fichero_test = open("entrada/problema_real2_no_etiquetados.txt", "r")
        entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test = leer3(fichero_entrenamiento, fichero_test)
    else:
        fichero = open("entrada/problema_real" + str(num_problema) + ".txt", "r")
        entradas_entrenamiento, salidas_entrenamiento, entradas_test, salidas_test = leer1(fichero, 0.7)
        if num_problema in [4, 6]:
            entradas_entrenamiento, entradas_test = normalizar(entradas_entrenamiento, entradas_test)

    # Añadimos el sesgo a los datos de entrada
    for i in range(len(entradas_entrenamiento)):
        entradas_entrenamiento[i].append(1)
    for i in range(len(entradas_test)):
        entradas_test[i].append(1)

    # Cambiamos salidas en binario a bipolar
    for i in range(len(salidas_entrenamiento)):
        for j in range(len(salidas_entrenamiento[i])):
            if salidas_entrenamiento[i][j] == 0:
                salidas_entrenamiento[i][j] = -1
    for i in range(len(salidas_test)):
        for j in range(len(salidas_test[i])):
            if salidas_test[i][j] == 0:
                salidas_test[i][j] = -1
    
    # Creamos la red
    red_neuronal = RedNeuronal()
    capa_entrada = Capa()
    capa_salida = Capa()
    capa_oculta = Capa()
    n_entradas = len(entradas_entrenamiento[0])
    n_salidas = len(salidas_entrenamiento[0])

    for i in range(n_entradas):
        if i != n_entradas-1:
            capa_entrada.anyadir(Neurona(0, "Entrada", "x" + str(i+1)))
        else:
            capa_entrada.anyadir(Neurona(0, "Entrada", "1"))

    for i in range(n_salidas):
        capa_salida.anyadir(Neurona(0, "Perceptron", "y" + str(i+1)))

    for i in range(p):
        capa_oculta.anyadir(Neurona(0, "Perceptron", "z"+str(i+1)))

    capa_entrada.conectar_capa(capa_oculta, -1, 1)
    capa_oculta.anyadir(Neurona(0, "Entrada", "1"))
    capa_oculta.conectar_capa(capa_salida, -1, 1)

    red_neuronal.anyadir(capa_entrada)
    red_neuronal.anyadir(capa_oculta)
    red_neuronal.anyadir(capa_salida)
    red_neuronal.inicializar()

    # Inicializamos variables y listas
    max_incremento_pesos = 1
    n_epoch = 0
    error_entrenamiento = []
    error_test = []
    tasa_entrenamiento = []
    tasa_test = []

    #Declaramos las matrices incW, incV y delta (capa salida)
    n = n_entradas
    m = n_salidas
    incW = np.zeros([p+1, m])
    incV = np.zeros([n, p])
    delta = np.zeros(m)

    # Hasta que la condicion de parada no se cumpla, entrenamos la red
    while max_incremento_pesos > 0 and n_epoch < max_epochs:
        print("Epoch:", n_epoch)
        for q in range(0, len(entradas_entrenamiento)):
            for j in range(n_entradas):
                entrada = entradas_entrenamiento[q][j]
                neurona = capa_entrada.neuronas[j]
                neurona.inicializar(entrada)

            red_neuronal.disparar()
            red_neuronal.propagar()
            red_neuronal.disparar()
            red_neuronal.propagar()
            capa_salida.disparar()

            # Calculamos los incrementos de los pesos
            for k in range(m):
                t_k = salidas_entrenamiento[q][k]
                y_k = capa_salida.neuronas[k].valor_salida
                y_in_k = capa_salida.neuronas[k].valor_entrada
                delta_k = (t_k - y_k)*f_prima(y_in_k)
                delta[k] = delta_k
                for j in range(p+1):
                    z_j = capa_oculta.neuronas[j].valor_salida
                    incW[j, k] = tasa*delta_k*z_j

            for j in range(p):
                w_j = []
                for conexion in capa_oculta.neuronas[j].conexiones:
                    w_j.append(conexion.peso)
                beta_in_j = np.dot(delta, np.array(w_j))
                z_in_j = capa_oculta.neuronas[j].valor_entrada
                beta_j = beta_in_j * f_prima(z_in_j)
                for i in range(n):
                    x_i = capa_entrada.neuronas[i].valor_salida
                    incV[i,j] = tasa*beta_j*x_i

            # Actualizamos los pesos
            for i in range(n):
                for j in range(p):
                    capa_entrada.neuronas[i].conexiones[j].peso += incV[i,j]
            for j in range(p+1):
                for k in range(m):
                    capa_oculta.neuronas[j].conexiones[k].peso += incW[j,k]     

        n_epoch += 1
        predicciones = calcular_predicciones(red_neuronal, entradas_entrenamiento)
        err = error_cuadratico(predicciones, salidas_entrenamiento)
        error_entrenamiento.append(err)
        tasa_entrenamiento.append(tasa_aciertos(predicciones, salidas_entrenamiento))
        if num_problema > 0:
            predicciones = calcular_predicciones(red_neuronal, entradas_test)
            err = error_cuadratico(predicciones, salidas_test)
            error_test.append(err)
            tasa_test.append(tasa_aciertos(predicciones, salidas_test))

    if num_problema > 0:
        plt.plot(range(len(error_entrenamiento)), error_entrenamiento, label="Entrenamiento")
        plt.plot(range(len(error_test)), error_test, label="Test")
        plt.legend()
        plt.title(f"ECM Problema {num_problema}: epochs={max_epochs}, alpha={tasa}, p={p}")
        plt.show()
        plt.plot(range(len(tasa_entrenamiento)), tasa_entrenamiento, label="Entrenamiento")
        plt.plot(range(len(tasa_test)), tasa_test, label="Test")
        plt.legend()
        plt.title(f"Tasa Aciertos Problema {num_problema}: epochs={max_epochs}, alpha={tasa}, p={p}")
        plt.show()

        m = calcular_matriz_confusion(predicciones, salidas_test)
        df = pd.DataFrame(m, columns=["Positivo", "Negativo"], index=["Positivo", "Negativo"])
        sn.heatmap(df, annot=True, cmap="Blues", vmin=0, vmax=1)
        plt.xlabel("Real")
        plt.ylabel("Predicción")
        plt.show()
    elif num_problema == 0:
        plt.plot(range(len(error_entrenamiento)), error_entrenamiento)
        plt.title(f"ECM Problema {num_problema}: epochs={max_epochs}, alpha={tasa}, p={p}")
        plt.show()
        plt.plot(range(len(tasa_entrenamiento)), tasa_entrenamiento)
        plt.title(f"Tasa Aciertos Problema {num_problema}: epochs={max_epochs}, alpha={tasa}, p={p}")
        plt.show()
        fichero_predicciones = open("predicciones/prediccion_problema_real2.txt", "w")
        predicciones = calcular_predicciones(red_neuronal, entradas_test)
        for prediccion in predicciones:
            if prediccion[0] > prediccion[1]:
                fichero_predicciones.write("1 -1\n")
            else:
                fichero_predicciones.write("-1 1\n")


if __name__ == "__main__":
    main()
    
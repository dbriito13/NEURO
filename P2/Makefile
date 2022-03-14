# McCullock Pitts
ayuda_mp:
	@echo "Los argumentos son ENTRADA y SALIDA, por ejemplo:"
	@echo "make ejecuta_mp_con ENTRADA=entrada/entrada.txt SALIDA=salida/salida.txt"
	@echo ""
	@echo "- ENTRADA: fichero de entrada."
	@echo "- SALIDA: fichero de salida."

compila_mp:
	@echo Usamos python, no hay que compilar.

ejecuta_mp:
	@python3 src/McCulloch-Pitts.py entrada/entrada.txt salida/salida.txt

ejecuta_mp_con:
	@python3 src/McCulloch-Pitts.py $(ENTRADA) $(SALIDA)


# Perceptron
ayuda_perceptron:
	@echo "Los argumentos son MAX_EPOCHS, UMBRAL y TASA, por ejemplo:"
	@echo "make ejecuta_perceptron_con MAX_EPOCHS=60 UMBRAL=0.25 TASA=0.02 P=2"
	@echo ""
	@echo "- MAX_EPOCHS: número máximo de epochs que entrena la red."
	@echo "- UMBRAL: umbral de activación de las neuronas."
	@echo "- TASA: tasa de aprendizaje."

ejecuta_perceptron:
	@python3 src/perceptron_problema_real.py 1 60 0.25 0.02 2

ejecuta_perceptron_con:
	@python3 src/perceptron_problema_real.py 1 $(MAX_EPOCHS) $(UMBRAL) $(TASA) $(P)


# Adaline
ayuda_adaline:
	@echo "Los argumentos son MAX_EPOCHS, TOLERANCIA y TASA, por ejemplo:"
	@echo "make ejecuta_adaline_con MAX_EPOCHS=60 TOLERANCIA=0.01 TASA=0.02"
	@echo ""
	@echo "- MAX_EPOCHS: número máximo de epochs que entrena la red."
	@echo "- TOLERANCIA: tolerancia usada para parar el entrenamiento."
	@echo "- TASA: tasa de aprendizaje."

ejecuta_adaline:
	@python3 src/adaline_problema_real.py 1 60 0.01 0.02

ejecuta_adaline_con:
	@python3 src/adaline_problema_real.py 1 $(MAX_EPOCHS) $(TOLERANCIA) $(TASA)


# Predicciones en el problema real 2
calcular_predicciones:
	@python3 src/perceptron_problema_real.py 2 60 0.25 0.02
	@python3 src/adaline_problema_real.py 2 60 0.01 0.02


# Problemas logicos
ejecuta_problemas_logicos:
	@python3 src/perceptron_problemas_logicos.py entrada/or.txt salida/salida_or.txt 50 0.2 1
	@python3 src/perceptron_problemas_logicos.py entrada/and.txt salida/salida_and.txt 50 0.2 1
	@python3 src/perceptron_problemas_logicos.py entrada/nand.txt salida/salida_nand.txt 50 0.2 1
	@python3 src/perceptron_problemas_logicos.py entrada/xor.txt salida/salida_xor.txt 50 0.2 1


# Necesario
compila:
	@echo Usamos python, no hay que compilar.



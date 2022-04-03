compilar:
	@echo Usamos python, no hay que compilar.

ayuda_retro:
	@echo "Los argumentos son NUM_PROB, MAX_EPOCHS, TASA y P, por ejemplo:"
	@echo "make ejecuta_perceptron_con NUM_PROB=1 MAX_EPOCHS=100 TASA=0.025 P=16"
	@echo ""
	@echo "- NUM_PROB: problema que se desea resolver."
	@echo "- MAX_EPOCHS: número máximo de epochs que entrena la red."
	@echo "- TASA: tasa de aprendizaje."
	@echo "- P: número de neuronas en la capa oculta."

ejecuta_perceptron_con:
	@python3 src/perceptron_problema_real.py $(NUM_PROB) $(MAX_EPOCHS) $(TASA) $(P)

ejecuta_problema_real1:
	@python3 src/perceptron_problema_real.py 1 200 0.01 4

ejecuta_problema_real2:
	@python3 src/perceptron_problema_real.py 2 100 0.025 16

ejecuta_problema_real3:
	@python3 src/perceptron_problema_real.py 3 150 0.01 8

ejecuta_problema_real4:
	@python3 src/perceptron_problema_real.py 4 150 0.001 16

ejecuta_problema_real5:
	@python3 src/perceptron_problema_real.py 5 175 0.01 32

ejecuta_problema_real6:
	@python3 src/perceptron_problema_real.py 6 10 0.1 20

calcular_predicciones:
	@python3 src/perceptron_problema_real.py 0 20 0.025 16

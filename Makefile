ayuda_mp:
	@echo "Los argumentos son ENTRADA y SALIDA, por ejemplo: make ejecuta_mp ENTRADA=entrada.txt SALIDA=salida.txt"

compila_mp:
	@echo Estamos en python, no hay que compilar.

ejecuta_mp:
	@python3 ejemplo.py $(ENTRADA) $(SALIDA)
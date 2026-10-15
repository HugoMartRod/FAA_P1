import numpy as np

class Clasificador:
    """
    Clase base para todos los algoritmos de Machine Learning. 
    Define el ciclo de vida y las operaciones obligatorias.
    """
    
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        raise NotImplementedError("El método de entrenamiento debe ser implementado por el algoritmo específico.")
    
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        raise NotImplementedError("El método de clasificación debe ser implementado por el algoritmo específico.")
    
    def error(self, datos, predicciones):
        """
        Calcula el porcentaje de fallos comparando las clases reales 
        con las predicciones devueltas por el modelo.
        """
        clases_reales = datos[:, -1]
        fallos = sum(clases_reales != predicciones)
        porcentaje_error = fallos / len(clases_reales)
        return porcentaje_error
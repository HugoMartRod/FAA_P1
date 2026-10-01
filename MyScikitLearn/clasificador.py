class Clasificador:
    """
    Clase base para todos los algoritmos de Machine Learning. 
    Define el ciclo de vida y las operaciones obligatorias.
    """
    
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        """
        Construye el modelo a partir de los datos de entrenamiento.
        Debe ser sobreescrito por las clases hijas.
        """
        raise NotImplementedError("El método de entrenamiento debe ser implementado por el algoritmo específico.")
    
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        """
        Evalúa los datos de prueba y devuelve las predicciones.
        Debe ser sobreescrito por las clases hijas.
        """
        raise NotImplementedError("El método de clasificación debe ser implementado por el algoritmo específico.")
    
    def error(self, datos, predicciones):
        """
        Calcula el porcentaje de fallos comparando las clases reales 
        con las predicciones devueltas por el modelo.
        Común a todos los clasificadores.
        """
        # Asumimos que la clase real es la última columna de los datos
        clases_reales = datos[:, -1]
        
        # Contamos cuántos fallos hay comparando los arrays
        fallos = sum(clases_reales != predicciones)
        
        # Devolvemos el porcentaje de error
        porcentaje_error = fallos / len(clases_reales)
        return porcentaje_error
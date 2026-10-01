import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, filepath):
        # Lee 2el fichero original
        self.datos_originales = pd.read_csv(filepath)
        
        self.diccionarios = {}
        self.datos_numericos = None
        
        self._construir_diccionarios()
        self._transformar_datos()

    def _construir_diccionarios(self):
        # Recorre cada atributo para crear el diccionario
        for columna in self.datos_originales.columns:
            if pd.api.types.is_numeric_dtype(self.datos_originales[columna]):
                # Si ya es numérico, diccionario vacío
                self.diccionarios[columna] = {}
            else:
                # Si es nominal, se ordenan las claves lexicográficamente y se asigna un entero
                valores_unicos = sorted(self.datos_originales[columna].dropna().unique())
                self.diccionarios[columna] = {valor: indice for indice, valor in enumerate(valores_unicos)}

    def _transformar_datos(self):
        # Crea la versión numérica usando los diccionarios generados
        self.datos_numericos = self.datos_originales.copy()
        for columna, mapeo in self.diccionarios.items():
            if mapeo:
                self.datos_numericos[columna] = self.datos_numericos[columna].map(mapeo)

    def obtener_datos(self, filas=None, columnas=None):
        # Devuelve los datos numéricos total o parcialmente
        datos = self.datos_numericos
        if filas is not None:
            datos = datos.iloc[filas, :]
        if columnas is not None:
            datos = datos.iloc[:, columnas]
        return datos.values
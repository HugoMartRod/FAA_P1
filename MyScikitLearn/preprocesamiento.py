import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, filepath):
        # Lee el fichero original
        self.datos_originales = pd.read_csv(filepath)

        self.diccionarios = {}
        self.nominalAtributos = []
        self.datos_numericos = None

        self._construir_diccionarios()
        self._transformar_datos()

    @property
    def atributosDiscretos(self):
        # Indica para cada atributo (sin la clase) si es nominal
        return self.nominalAtributos[:-1]

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
            self.nominalAtributos.append(bool(self.diccionarios[columna]))

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

    def estandarizarDatos(self, media=True, std=True):
        """
        Estandariza los atributos numéricos de los datos con el mismo criterio
        que StandardScaler de Scikit-Learn. Ignora los atributos nominales
        (codificados con el diccionario) y la última columna (clase).
        """
        # Solución al LossySetitemError: Convertir todo a decimales (float)
        # antes de intentar sobrescribir con resultados de divisiones
        self.datos_numericos = self.datos_numericos.astype(float)

        columnas = self.datos_numericos.columns[:-1]

        for columna, es_nominal in zip(columnas, self.atributosDiscretos):
            if es_nominal:
                continue
            valores = self.datos_numericos[columna]

            valor_media = valores.mean() if media else 0.0
            # ddof=0 (desviación típica poblacional), igual que StandardScaler
            valor_std = valores.std(ddof=0) if std else 1.0

            # StandardScaler deja sin escalar las columnas constantes
            if valor_std == 0:
                valor_std = 1.0

            self.datos_numericos[columna] = (valores - valor_media) / valor_std
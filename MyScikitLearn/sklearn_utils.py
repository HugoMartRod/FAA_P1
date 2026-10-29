import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold


def estrategias_sklearn(semilla=None, numero_ejecuciones=10, proporcion_test=0.3, numero_particiones=5):
    """
    Reproduce con Scikit-Learn el mismo diseño experimental que nuestras
    estrategias de particionado: validación simple repetida y validación cruzada.
    """
    return {
        "Validación Simple": ShuffleSplit(n_splits=numero_ejecuciones, test_size=proporcion_test, random_state=semilla),
        "Validación Cruzada": KFold(n_splits=numero_particiones, shuffle=True, random_state=semilla)
    }


def evaluar_modelo_sklearn(crear_modelo, X, y, particionado):
    """
    Entrena y evalúa un modelo nuevo por cada partición generada.
    Devuelve el error medio y la desviación típica (en %), o el error
    capturado si el modelo no puede aplicarse a los datos.
    """
    errores = []
    try:
        for indices_train, indices_test in particionado.split(X):
            modelo = crear_modelo()
            modelo.fit(X[indices_train], y[indices_train])
            predicciones = modelo.predict(X[indices_test])
            errores.append(np.mean(predicciones != y[indices_test]))
    except Exception as e:
        return np.nan, np.nan, f"{type(e).__name__}: {e}"

    return np.mean(errores) * 100, np.std(errores) * 100, ""

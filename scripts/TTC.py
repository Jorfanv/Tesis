import pandas as pd
import ast
from copy import deepcopy
from collections import defaultdict

class TTCReubicacion:
    """
    Implementación del algoritmo Top Trading Cycles (TTC)
    adaptado para la reubicación de colaboradores en tiendas
    según sus preferencias de cercanía.
    """

    def __init__(self, df):
        # Validación básica
        if not all(col in df.columns for col in ["NUMERO DE IDENTIFICACION", "LUGAR DE TRABAJO", "distancias_cercanos"]):
            raise ValueError("El DataFrame no tiene las columnas necesarias.")

        # Preparación de datos
        self.df = df.copy()
        self.df["distancias_cercanos"] = self.df["distancias_cercanos"].apply(ast.literal_eval)
        self.df["preferencias"] = self.df["distancias_cercanos"].apply(lambda d: list(d.keys()))

        # Estructuras de asignación
        self.empleado_a_tienda = dict(zip(self.df["NUMERO DE IDENTIFICACION"], self.df["LUGAR DE TRABAJO"]))
        self.tienda_a_empleados = defaultdict(list)
        for emp, tienda in self.empleado_a_tienda.items():
            self.tienda_a_empleados[tienda].append(emp)

        # Copias para proceso iterativo
        self.empleado_a_tienda_final = deepcopy(self.empleado_a_tienda)
        self.tienda_a_empleados_final = deepcopy(self.tienda_a_empleados)

        # Resultados
        self.asignaciones_finales = {}
        self.ciclos_encontrados = []
        self.rondas = []

    def ejecutar(self, verbose=True):
        ronda = 1
        while self.empleado_a_tienda_final:
            if verbose:
                print(f"\n🟦 RONDA {ronda} → Colaboradores activos: {len(self.empleado_a_tienda_final)}")

            # Paso 1: cada empleado apunta a su mejor tienda disponible
            apunta_a = {}
            for emp, prefs in self.df.set_index("NUMERO DE IDENTIFICACION")["preferencias"].items():
                if emp in self.empleado_a_tienda_final:
                    prefs_disponibles = [t for t in prefs if t in self.tienda_a_empleados_final]
                    apunta_a[emp] = prefs_disponibles[0] if prefs_disponibles else self.empleado_a_tienda_final[emp]

            # Paso 2: cada tienda apunta a un empleado actual
            tienda_apunta = {}
            for tienda, empleados in self.tienda_a_empleados_final.items():
                if empleados:
                    tienda_apunta[tienda] = empleados[0]

            # Paso 3: buscar ciclos
            visitados = set()
            ciclo_en_ronda = False

            while True:
                ciclo = []
                emp = next((e for e in self.empleado_a_tienda_final if e not in visitados), None)
                if emp is None:
                    break

                while emp not in ciclo and emp in apunta_a:
                    ciclo.append(emp)
                    visitados.add(emp)
                    tienda = apunta_a[emp]
                    if tienda not in tienda_apunta:
                        break
                    emp = tienda_apunta[tienda]

                if emp in ciclo:
                    inicio = ciclo.index(emp)
                    ciclo = ciclo[inicio:]
                    ciclo_en_ronda = True
                    self.ciclos_encontrados.append(ciclo)

                    if verbose:
                        print(f"🔁 Ciclo encontrado: {' → '.join(['Empleado ' + str(e) for e in ciclo])}")

                    # Asignar y eliminar del sistema
                    for e in ciclo:
                        tienda_asignada = apunta_a[e]
                        self.asignaciones_finales[e] = tienda_asignada
                        if e in self.empleado_a_tienda_final:
                            del self.empleado_a_tienda_final[e]
                        tienda_actual = self.empleado_a_tienda[e]
                        if e in self.tienda_a_empleados_final[tienda_actual]:
                            self.tienda_a_empleados_final[tienda_actual].remove(e)

                    # Eliminar tiendas vacías
                    tiendas_vacias = [t for t, emps in self.tienda_a_empleados_final.items() if len(emps) == 0]
                    for t in tiendas_vacias:
                        del self.tienda_a_empleados_final[t]

                    # Guardar info de la ronda
                    self.rondas.append({
                        "ronda": ronda,
                        "num_colaboradores": len(apunta_a),
                        "num_ciclos": len(self.ciclos_encontrados)
                    })

                    break  # Recalcular en el nuevo sistema

            if not ciclo_en_ronda:
                if verbose:
                    print("⚠️ No se encontraron más ciclos.")
                break

            ronda += 1

        if verbose:
            print("\n✅ Proceso completado. Ciclos totales encontrados:", len(self.ciclos_encontrados))

        return self._construir_resultados()

    # ----------------------------------------------------------------------
    def _construir_resultados(self):
        """Crea los DataFrames de resultados"""
        df_res = self.df.copy()
        df_res["tienda_recomendada"] = df_res["NUMERO DE IDENTIFICACION"].map(self.asignaciones_finales)
        df_res["ciclo"] = None

        # Asignar ciclo correspondiente
        for i, ciclo in enumerate(self.ciclos_encontrados, 1):
            for e in ciclo:
                df_res.loc[df_res["NUMERO DE IDENTIFICACION"] == e, "ciclo"] = i

        # Calcular distancia y mejora
        def distancia_recomendada(row):
            dic = row["distancias_cercanos"]
            tienda = row["tienda_recomendada"]
            if isinstance(dic, dict) and tienda in dic:
                return dic[tienda]["distancia_km"]
            return row["distancia_actual_km"]

        def tiempo_recomendado(row):
            dic = row["distancias_cercanos"]
            tienda = row["tienda_recomendada"]
            if isinstance(dic, dict) and tienda in dic:
                return dic[tienda]["tiempo_min"]
            return row["tiempo_actual_min"]

        df_res["distancia_recomendada_km"] = df_res.apply(distancia_recomendada, axis=1)
        df_res["tiempo_recomendado_min"] = df_res.apply(tiempo_recomendado, axis=1)
        df_res["mejora_%_distancia"] = (
            (df_res["distancia_actual_km"] - df_res["distancia_recomendada_km"]) /
            df_res["distancia_actual_km"]
        ) * 100

        # Resumen de ciclos
        resumen_ciclos = pd.DataFrame([
            {"ciclo": i + 1, "empleados": ciclo, "num_empleados": len(ciclo)}
            for i, ciclo in enumerate(self.ciclos_encontrados)
        ])

        return df_res, resumen_ciclos

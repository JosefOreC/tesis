#!/usr/bin/env python3
"""
calcular_cvc.py

Calcula el Coeficiente de Validez de Contenido (CVC) según Hernández-Nieto a partir
de un CSV con las puntuaciones de jueces.

Entrada CSV (ejemplo):
Item,Juez 01,Juez 02,Juez 03,Juez 04
1,17,20,20,20
2,20,20,20,20
...

Salida:
- Archivo CSV con columnas: Item, mean, std, n_jueces, CVC, Pe, CVC_corr
- Resumen impreso en consola
"""

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path

def interpretar_args():
    p = argparse.ArgumentParser(description="Calcular CVC (Hernández-Nieto) desde un CSV de jueces.")
    p.add_argument("--csv", "-c", required=True, help="Ruta al archivo CSV con las puntuaciones.")
    p.add_argument("--vmax", "-v", type=float, default=None,
                   help="Valor máximo de la escala (por ejemplo 4, 5, 20). Si no se indica, se usa el valor máximo observado en los datos.")
    p.add_argument("--out", "-o", default="cvc_resultados.csv", help="Nombre del archivo CSV de salida (por defecto cvc_resultados.csv).")
    p.add_argument("--clip", action="store_true", help="Clippear CVC corregido a [0,1] (por defecto True).")
    return p.parse_args()

def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)
    if df.shape[1] < 2:
        raise ValueError("El CSV debe contener al menos dos columnas: 'Item' y al menos un juez.")
    return df

def detectar_columnas_jueces(df):
    # Suponemos que la primera columna es 'Item' o identificador; el resto son jueces
    cols = list(df.columns)
    item_col = cols[0]
    judge_cols = cols[1:]
    return item_col, judge_cols

def calcular_cvc(df, item_col, judge_cols, vmax=None, clip=True):
    # Extraer matriz de puntuaciones
    scores = df[judge_cols].astype(float)
    n_jueces = scores.shape[1]
    n_items = scores.shape[0]
    # Determinar Vmax
    vmax_used = vmax if vmax is not None else scores.max().max()
    if vmax_used <= 0:
        raise ValueError("Vmax debe ser > 0.")
    # Cálculos por ítem
    mean = scores.mean(axis=1)
    std = scores.std(axis=1, ddof=0)  # ddof=0 (desviación poblacional) o ddof=1 (muestral). Hernández-Nieto usa desviación estándar simple; uso ddof=0 por consistencia con Pe.
    # Si prefieres ddof=1 (muestral), reemplaza ddof=0 por ddof=1.
    cvc = mean / vmax_used
    pe = std / vmax_used
    cvc_corr = cvc - pe
    if clip:
        cvc_corr = cvc_corr.clip(lower=0.0, upper=1.0)
    # Construir DataFrame de resultados
    resultados = pd.DataFrame({
        item_col: df[item_col],
        "mean": mean,
        "std": std,
        "n_jueces": n_jueces,
        "Vmax_usado": vmax_used,
        "CVC": cvc,
        "Pe": pe,
        "CVC_corr": cvc_corr
    })
    # CVC total (promedio de CVC corregidos)
    cvc_total = resultados["CVC_corr"].mean()
    return resultados, cvc_total, n_jueces, n_items, vmax_used

def interpretar_cvc(valor):
    # Interpretación orientativa
    if valor >= 0.90:
        return "Excelente (>= 0.90)"
    elif valor >= 0.80:
        return "Buena (0.80 - 0.89)"
    elif valor >= 0.70:
        return "Aceptable (0.70 - 0.79)"
    else:
        return "No válida (< 0.70) - revisar ítems"

def main():
    args = interpretar_args()
    ruta = Path(args.csv)
    if not ruta.exists():
        print(f"ERROR: No existe el archivo {ruta}", file=sys.stderr)
        sys.exit(1)

    df = cargar_datos(ruta)
    item_col, judge_cols = detectar_columnas_jueces(df)

    resultados, cvc_total, n_jueces, n_items, vmax_used = calcular_cvc(
        df, item_col, judge_cols, vmax=args.vmax, clip=args.clip
    )

    # Guardar resultados
    resultados.to_csv(args.out, index=False)
    # Imprimir resumen
    print("=== Resumen Cálculo CVC Hernández-Nieto ===")
    print(f"Archivo leído: {ruta}")
    print(f"Número de ítems: {n_items}")
    print(f"Número de jueces detectados: {n_jueces}")
    print(f"Vmax usado: {vmax_used}")
    print(f"CVC total (promedio CVC corregidos): {cvc_total:.4f} -> {interpretar_cvc(cvc_total)}")
    print(f"Resultados guardados en: {args.out}")
    print("\nPrimeras filas del resultado:")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(resultados.head(10).to_string(index=False))

if __name__ == "__main__":
    main()

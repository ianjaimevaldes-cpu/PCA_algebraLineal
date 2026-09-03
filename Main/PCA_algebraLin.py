import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def leer_datos():
    rd = pd.read_csv("Data/Datos_SP2025.csv")
    print(rd.head())
    #se seleccionan las columnas
    X = rd[["ind1", "ind2", "ind3"]]
    #se estandariza, aunque checare primero estandarizado y luego sin estandarizar
    #a ver qie tanto cambia
    X_estand = (X-X.mean())/X.std()

    #Se saca la matriz de covarianzas
    S = X_estand.cov()
    #S = X.cov()
    print("\n matriz de covarianzas \n", S)

    #sacar autovalores y autovectores

    autovalores, autovectores= np.linalg.eigh(S)
    autovalores = np.flip(autovalores)
    autovectores = np.flip(autovectores, axis = 1)
    print("\n autovalores : \n", autovalores)
    print("\n Autovectores: \n", autovectores)

    suma_autov = autovalores.sum()

    porcentajes = (autovalores / suma_autov)* 100
    print(porcentajes)


    #scree plot
    
    componentes = range(1, len(autovalores)+1)

    plt.figure()
    plt.plot(componentes, autovalores, marker="o")
    plt.axhline(y=1, color="orange", linestyle="--")
    plt.xlabel("Componente Principal")
    plt.ylabel("autovalor")
    plt.title("Scree Plot")
    plt.xticks(componentes, [f"PC{i}"for i in componentes])
    plt.grid(alpha=0.3)

    plt.savefig("Data/ScreePlot-png", dpi =150, bbox_inches="tight")

    PC1 = X_estand @ autovectores[:, 0]
    PC2 = X_estand @ autovectores[:, 1]
    PC3 = C_estand @ autovectores[:, 3]

    #graficar

    

    plt.scatter(PC1, PC2)

    plt.xlabel(f"PC1 ({porcentajes[0]:.2f}% de varianza")
    plt.ylabel(f"PC2 ({porcentajes[1]:.2f}% de varianza")
    plt.title("Grafica de datos en 2D usando reduccion de dimencionalidad con PCA")

    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.grid(alpha=0.3)

    plt.savefig("Data/Grafica_PCA.png", dpi=150, bbox_inches="tight")

    

leer_datos()

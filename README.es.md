<p align="right"><a href="README.md">English</a> · <b>Español</b></p>

# Regresión lineal desde cero con NumPy: ¿puede un plano predecir la profundidad de los sismos de Ecuador?

Regresión lineal múltiple sin scikit-learn sobre 15 años de sismos del USGS: derivación de la función de costo, demostración de existencia y unicidad del mínimo, solución analítica, descenso por gradiente y verificación contra scikit-learn.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy) ![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas) ![scikit-learn](https://img.shields.io/badge/scikit--learn-solo%20verificaci%C3%B3n-F7931E?logo=scikitlearn&logoColor=white) ![Licencia](https://img.shields.io/badge/licencia-MIT-green) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eduardo0602/regresion-lineal-numpy-desde-cero/blob/main/notebooks/01_exploracion_datos_sismicos.ipynb)

## El problema

En Ecuador, la placa de Nazca se hunde bajo la Sudamericana (zona de Wadati–Benioff): hacia el este, los sismos deberían ser más profundos. ¿Basta la posición geográfica para predecir la profundidad del hipocentro con un modelo lineal? El proyecto responde implementando la regresión desde cero, con cada fórmula derivada y cada identidad demostrada.

## Datos

| Característica | Detalle |
|---|---|
| Fuente | [USGS Earthquake Hazards Program, FDSNWS Event Web Service](https://earthquake.usgs.gov/fdsnws/event/1/) |
| Consulta | Sismos en Ecuador, 2010–2025, magnitud ≥ 2 (rectángulo: N 2,5; S −6; O −82,5; E −74,5) |
| Dimensiones | 1 187 filas × 22 columnas (procesado: 1 187 × 3) |
| Variable objetivo | `depth`: profundidad del hipocentro (km) |
| Variables seleccionadas | `longitude` (Pearson 0,521) y `latitude` (Pearson −0,207); `mag` descartada por correlación prácticamente nula |

Los datos no se incluyen; se descargan con la URL de la API (ver Cómo reproducir).

## Fundamento matemático

El modelo minimiza

```math
J(\boldsymbol{\beta}) = \frac{1}{2m}\|X\boldsymbol{\beta} - \mathbf{y}\|^2,
```

con $`X \in \mathbb{R}^{1187 \times 3}`$ (columna de unos para el intercepto) y $`\boldsymbol{\beta} \in \mathbb{R}^3`$.

**Existencia y unicidad.** $`J`$ es continua y coerciva (por la descomposición espectral de $`X^\top X`$ y la cota con $`\lambda_{\min}`$), así que el teorema de Weierstrass garantiza un mínimo. La hessiana $`H_J = \frac{1}{m}X^\top X`$ es definida positiva si las columnas de $`X`$ son linealmente independientes, lo que da convexidad estricta y por tanto unicidad.

**Dos vías de solución.**
- Ecuaciones normales $`X^\top X\,\boldsymbol{\beta} = X^\top \mathbf{y}`$, resueltas con `np.linalg.solve` (no con `np.linalg.inv`, por estabilidad y eficiencia).
- Descenso por gradiente, $`\boldsymbol{\beta}_{t+1} = \boldsymbol{\beta}_t - \frac{\alpha}{m}X^\top(X\boldsymbol{\beta}_t - \mathbf{y})`$, derivado de la aproximación de Taylor de primer orden y la desigualdad de Cauchy–Schwarz.

## Resultados

- **Coeficientes (variables estandarizadas):** $`\beta_0 = 62{,}149`$ km, $`\beta_1 = 27{,}921`$ km por desviación estándar de longitud, $`\beta_2 = -9{,}347`$ km por desviación estándar de latitud.
- **$`\beta_1 > 0`$ confirma la geometría de Wadati–Benioff:** hacia el este la profundidad aumenta.
- **Ajuste limitado y reportado tal cual:** $`R^2 = 0{,}3000`$, RMSE $`= 45{,}94`$ km, MAE $`= 34{,}61`$ km. La profundidad es bimodal (sismos corticales de 0–30 km y de subducción de 100–200 km) y un único plano no la captura.
- **Buen condicionamiento:** los valores propios de $`X^\top X`$ son 1101,70; 1187,00 y 1272,30, con número de condición 1,15.

![Longitud frente a profundidad](reports/figures/longitude_vs_depth_regresion.png)

![Efecto de la tasa de aprendizaje](reports/figures/efecto_tasa_aprendizaje.png)

## Verificación

La solución analítica difiere de `LinearRegression` de scikit-learn en $`2{,}84 \times 10^{-13}`$ (máxima diferencia entre coeficientes). El descenso por gradiente llega a $`2{,}48 \times 10^{-4}`$ del óptimo en 118 épocas con $`\alpha = 0{,}1`$.

## Cómo reproducir

```bash
git clone https://github.com/Eduardo0602/regresion-lineal-numpy-desde-cero.git
cd regresion-lineal-numpy-desde-cero
conda create -n ds_portafolio python=3.11 -y
conda activate ds_portafolio
pip install -r requirements.txt
# Descargar los datos del USGS y guardarlos en data/raw/:
# https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2010-01-01&endtime=2025-12-31&minmagnitude=2&minlatitude=-6&maxlatitude=2.5&minlongitude=-82.5&maxlongitude=-74.5
jupyter lab   # abrir 01 y 02 en orden: Kernel → Restart & Run All
```

| Notebook | Contenido |
|---|---|
| [`01_exploracion_datos_sismicos.ipynb`](notebooks/01_exploracion_datos_sismicos.ipynb) | Distribución bimodal de la profundidad, Pearson frente a Spearman, selección de variables, corte transversal de Wadati–Benioff |
| [`02_regresion_lineal.ipynb`](notebooks/02_regresion_lineal.ipynb) | Función de costo, gradiente con identidades demostradas, existencia y unicidad; clase `RegresionLineal` (analítica y descenso por gradiente); verificación contra scikit-learn; efecto de $`\alpha`$ |

## Estructura del proyecto

```
regresion-lineal-numpy-desde-cero/
├── data/processed/       # sismos_clean.csv (1187 × 3), se genera con el notebook 01
├── notebooks/            # 01 → 02
├── reports/figures/      # 9 figuras
├── requirements.txt
└── LICENSE
```

## Limitaciones

- Un modelo lineal no captura la distribución bimodal de la profundidad; un modelo por régimen sísmico (cortical y de subducción) o uno no lineal sería el siguiente paso.
- Las métricas se calculan sobre todo el conjunto: el objetivo fue la implementación y su verificación, no la capacidad predictiva fuera de muestra.

## Lo que aprendí

1. **Demostrar, no declarar.** El mínimo único no existe porque lo diga un libro: se construye la cadena continuidad y coercividad, existencia, hessiana definida positiva, convexidad estricta, unicidad.
2. **La implementación no es la fórmula traducida.** $`(X^\top X)^{-1}X^\top \mathbf{y}`$ no se calcula invirtiendo la matriz: resolver el sistema es más estable y eficiente.
3. **Estandarizar no es opcional para el descenso por gradiente.** Con escalas distintas las curvas de nivel se deforman y el descenso oscila; estandarizar fue la diferencia entre converger en 118 épocas y no converger en 500.
4. **Un $`R^2`$ bajo también es información.** El 30 % de varianza explicada revela una estructura bimodal que un plano no puede capturar.

---

### Portafolio *De Matemático a Data Scientist*

| Proyecto | Pregunta | Herramientas |
|---|---|---|
| [Muestreo complejo con Ser Estudiante](https://github.com/Eduardo0602/muestreo-complejo-ser-estudiante/blob/main/README.es.md) | ¿Cuánto se equivoca quien ignora el diseño muestral? | R, survey |
| [EDA con datos sucios: defunciones 2021](https://github.com/Eduardo0602/eda-limpieza-defunciones-ecuador-pandas-sql/blob/master/README.es.md) | ¿Qué hay que corregir antes de confiar en un registro oficial? | Python, pandas, SQL |
| **Regresión lineal desde cero** (este repositorio) | ¿Puede un plano predecir la profundidad de los sismos de Ecuador? | Python, NumPy |
| [Álgebra lineal visual](https://github.com/Eduardo0602/algebra-lineal-visual-numpy/blob/main/README.es.md) | ¿Qué hace geométricamente una matriz? | Python, NumPy |

Eduardo Araque · Matemático (Universidad Central del Ecuador) · [GitHub](https://github.com/Eduardo0602) · [LinkedIn](https://www.linkedin.com/in/eduardo-araque-j%C3%A1come-311b93235)

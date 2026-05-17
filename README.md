# alzheimer-mri-ml

# Detección temprana del Alzheimer mediante machine learning sobre biomarcadores de neuroimagen estructural

Este repositorio contiene el pipeline completo de procesado, extracción de características y clasificación supervisada para la diferenciación entre sujetos **Cognitivamente Normales (CN)**, con **Deterioro Cognitivo Leve (MCI)** y con **Enfermedad de Alzheimer (AD)** a partir de imágenes de resonancia magnética estructural (sMRI).

Los datos empleados proceden de la base de datos pública **ADNI (Alzheimer's Disease Neuroimaging Initiative)**. Su uso está sujeto a los términos del acuerdo de uso de datos de ADNI, que **prohíbe expresamente la publicación de imágenes de pacientes y datasets completos**. Por este motivo, el repositorio no incluye datos crudos ni imágenes; únicamente contiene el código necesario para reproducir el análisis a partir de una descarga propia de ADNI.

> Para acceder a los datos originales es necesario registrarse y solicitar acceso en [adni.loni.usc.edu](https://adni.loni.usc.edu)

---

## Tabla de contenidos

- [Estructura del repositorio](#estructura-del-repositorio)
- [Flujo de trabajo](#flujo-de-trabajo)
- [Notebooks](#notebooks)
  - [1. Conversión a formato BIDS](#1-conversión-a-formato-bids)
  - [2. Extracción de features](#2-extracción-de-features)
  - [3. Clasificación CN vs AD](#3-clasificación-cn-vs-ad)
  - [4. Análisis de hiperparámetros y clasificación MCI vs AD](#4-análisis-de-hiperparámetros-y-clasificación-mci-vs-ad)
- [Atlas Harvard-Oxford](#atlas-harvard-oxford)
- [Requisitos del entorno](#requisitos-del-entorno)
  - [Librerías Python](#librerías-python)
  - [Docker — Instalación general](#docker--instalación-general)
  - [Docker — HeudiConv](#docker--heudiconv-notebook-1)
  - [Docker — fMRIPrep](#docker--fmriprep-preprocesado-de-imágenes)
- [Datos y licencia ADNI](#datos-y-licencia-adni)

---

## Estructura del repositorio

```
alzheimer-ml-neuroimaging
│
├── ADNI Data Use Agreement/
│   └── adni_data_use_agreement.pdf          # Condiciones legales de uso de los datos ADNI
│
├── Atlas Harvard Oxford/
│   ├── HarvardOxford-cort-maxprob-thr25-1mm.nii.gz    # Atlas cortical (plantillas)
│   ├── HarvardOxford-cort-maxprob-thr25-1mm.txt        # Atlas cortical (etiquetas)
│   ├── HarvardOxford-sub-maxprob-thr25-1mm.nii.gz     # Atlas subcortical (plantillas)
│   └── HarvardOxford-sub-maxprob-thr25-1mm.txt         # Atlas subcortical (etiquetas)
│
├── Conversión a formato BIDS/
│   └── conversion_bids.ipynb                # Notebook 1 — DICOM → BIDS con HeudiConv
│
├── Extraccion features/
│   └── extraccion_features.ipynb            # Notebook 2 — biomarcadores FreeSurfer + fMRIPrep
│
├── Clasificación CN vs AD/
│   └── clasificacion_cn_vs_ad.ipynb         # Notebook 3 — Elastic Net + Random Forest
│
├── Analisis hiperparámetros y clasificación MCI vs AD/
│   └── clasificacion_mci_vs_ad.ipynb        # Notebook 4 — Optuna + GridSearch + DCA
│
└── README.md
```

---

## Flujo de trabajo

El pipeline sigue un orden secuencial estricto. Cada etapa genera los archivos de entrada que consume la siguiente.

```
                    Imágenes DICOM (ADNI)
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  1. Conversión a formato BIDS                                │
│     Conversión a formato BIDS/conversion_bids.ipynb          │
│     DICOM → NIfTI + estructura BIDS (HeudiConv + Docker)     │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  2. Extracción de features                                   │
│     Extraccion features/extraccion_features.ipynb            │
│     fMRIPrep + FreeSurfer → CSVs de biomarcadores por        │
│     grupo (AD, EMCI, LMCI, CN) y sexo (m, f)                 │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  3. Clasificación CN vs AD                                   │
│     Clasificación CN vs AD/clasificacion_cn_vs_ad.ipynb      │
│     Elastic Net + Random Forest + GridSearchCV               │
│     → Selección de ROIs discriminativas                      │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  4. Análisis de hiperparámetros y clasificación MCI vs AD    │
│     Analisis hiperparámetros y clasificación MCI vs AD/      │
│     clasificacion_mci_vs_ad.ipynb                            │
│     Optuna (10.000 trials) + GridSearch refinado + DCA       │
└──────────────────────────────────────────────────────────────┘
```

---

## Notebooks

### [1. Conversión a formato BIDS](https://github.com/joseluis004/alzheimer-mri-ml/blob/main/Conversi%C3%B3n%20a%20formato%20BIDS/de_dicom_a_nifti_y_BIDS.ipynb)

Convierte las imágenes descargadas de ADNI en formato DICOM a la estructura estándar **BIDS (Brain Imaging Data Structure)**, requisito previo para el preprocesado con fMRIPrep.

El proceso se automatiza mediante **HeudiConv** ejecutado sobre Docker:

- Creación del fichero heurístico para mapear series MPRAGE a `sub-{subject}/anat/sub-{subject}_T1w`
- Detección automática de sujetos en el directorio DICOM
- Conversión en bucle para todos los sujetos con `dcm2niix`
- Limpieza de archivos residuales generados automáticamente por HeudiConv (`.heudiconv`, `derivatives`, `CHANGES`, `README`, `.bidsignore`)

**Entrada:** carpeta raíz con archivos DICOM organizados por sujeto  
**Salida:** estructura BIDS con archivos `.nii.gz` y `.json` por sujeto  
**Requisitos:** Docker con imagen `nipy/heudiconv:latest`

---

### [2. Extracción de características](https://github.com/joseluis004/alzheimer-mri-ml/blob/main/Extracción%20características/Extracción%20características.ipynb)

Extrae los biomarcadores morfométricos cerebrales a partir de las salidas de **fMRIPrep** y **FreeSurfer**. Contiene cuatro pipelines de extracción independientes que generan un CSV por grupo (`AD`, `EMCI`, `LMCI`, `CN`) y sexo (`m`, `f`):

| Pipeline | Fuente | Características extraídas | Normalización |
|---|---|---|---|
| Volúmenes por ROI (Harvard-Oxford) | fMRIPrep `probseg` | GM, WM, CSF por región cortical y subcortical | Sí, por ICV |
| Grosores corticales | FreeSurfer `aparc.stats` | `ThickAvg` por región y hemisferio | No |
| Volúmenes subcorticales | FreeSurfer `aseg.stats` | Volumen por estructura + ICV | Sí, por ICV |
| Dataset combinado | FreeSurfer `aseg` + `aparc` | Volúmenes + grosores + área superficial | Parcial |

Los atlas empleados están disponibles en [`Atlas Harvard Oxford/`](./Atlas%20Harvard%20Oxford/).

**Entrada:** salidas de fMRIPrep (`probseg`) y FreeSurfer (`stats/`)  
**Salida:** `volumenes_grosores_corticales_y_surfarea_{grupo}_{genero}.csv`
**Requisitos:** nibabel, nilearn, pandas, numpy

---

### [3. Clasificación CN vs AD](https://github.com/joseluis004/alzheimer-mri-ml/blob/main/Clasificaci%C3%B3n%20CN%20vs%20AD/CN_vs_AD.ipynb)

Implementa el pipeline de clasificación binaria **CN vs AD** y la selección de las ROIs más discriminativas que se transferirán al problema MCI vs AD.

El pipeline está compuesto por tres etapas encadenadas en un `sklearn.Pipeline`:

1. `StandardScaler` — estandarización de variables
2. `SelectFromModel` con Elastic Net — selección automática de ROIs mediante regularización L1+L2
3. `RandomForestClassifier` — clasificador final con `class_weight='balanced'`

La búsqueda de hiperparámetros se realiza con `GridSearchCV` y validación cruzada estratificada de 5 folds, optimizando la **Balanced Accuracy** y evaluando simultáneamente F1, ROC-AUC, Precision, Recall y Specificity. Tras el entrenamiento se genera un ranking completo de todas las ROIs ordenadas por valor absoluto del coeficiente Elastic Net.

**Entrada:** CSVs generados en el notebook anterior  
**Salida:** `feature_names` (lista de ROIs seleccionadas)
**Requisitos:** scikit-learn, pandas, numpy, seaborn, matplotlib

---

### [4. Análisis de hiperparámetros y clasificación MCI vs AD](https://github.com/joseluis004/alzheimer-mri-ml/blob/main/Analisis%20hiperpar%C3%A1metros%20y%20clasificaci%C3%B3n%20MCI%20vs%20AD/Analisis_hiperparámetros_y_clasificación_MCI_vs_AD.ipynb)

Aplica las ROIs seleccionadas en el notebook anterior al problema más difícil del espectro: la separación entre **MCI** (EMCI + LMCI) y **AD**. El notebook se estructura en tres bloques:

**Bloque 1 — Optimización bayesiana con Optuna**

Búsqueda de hiperparámetros con el muestreador **TPE** (*Tree-structured Parzen Estimator*) multivariante sobre 10.000 trials. Tras la optimización se analiza el top 1% de los trials para identificar los rangos óptimos de cada hiperparámetro mediante estadísticas descriptivas.

| Hiperparámetro | Rango explorado |
|---|---|
| `n_estimators` | [1, 1000] |
| `max_depth` | [1, 30] |
| `min_samples_split` | [2, 30] |
| `min_samples_leaf` | [1, 30] |

**Bloque 2 — GridSearch refinado**

Grid de búsqueda acotado sobre los rangos óptimos del top 1% de Optuna (`n_estimators` [350, 470], `max_depth` [10, 25], `min_samples_split` [24, 30], `min_samples_leaf` [1, 2]), con evaluación de seis métricas en validación cruzada estratificada de 5 pliegues.

**Bloque 3 — Ajuste de umbral y Curva de Decisión Clínica (DCA)**

Ajuste del umbral de decisión sobre predicciones *out-of-fold* con restricción $\text{Recall}_{AD} \geq 0.75$, para priorizar la detección de AD frente a la precisión global. La **Curva de Decisión Clínica** evalúa el beneficio neto del modelo frente a las estrategias extremas de tratar a todos o no tratar a nadie:

**Entrada:** `feature_names` del notebook 3 + CSVs de grupos MCI y AD  
**Salida:** modelo final evaluado con métricas CV y test + visualizaciones  
**Requisitos:** scikit-learn, optuna, pandas, numpy, seaborn, matplotlib

---

## Atlas Harvard-Oxford

Los atlas empleados en la extracción de volúmenes por ROI están disponibles en [`Atlas Harvard Oxford/`](./Atlas%20Harvard%20Oxford/):

| Archivo | Descripción |
|---|---|
| [`HarvardOxford-cort-maxprob-thr25-1mm.nii.gz`](./Atlas%20Harvard%20Oxford/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz) | Atlas cortical — imagen de regiones |
| [`HarvardOxford-cort-maxprob-thr25-1mm.txt`](./Atlas%20Harvard%20Oxford/HarvardOxford-cort-maxprob-thr25-1mm.txt) | Atlas cortical — nombres de regiones |
| [`HarvardOxford-sub-maxprob-thr25-1mm.nii.gz`](./Atlas%20Harvard%20Oxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz) | Atlas subcortical — imagen de regiones |
| [`HarvardOxford-sub-maxprob-thr25-1mm.txt`](./Atlas%20Harvard%20Oxford/HarvardOxford-sub-maxprob-thr25-1mm.txt) | Atlas subcortical — nombres de regiones |

Los atlas están disponibles públicamente como parte de **FSL (FMRIB Software Library)**. Para más información: [fsl.fmrib.ox.ac.uk](https://fsl.fmrib.ox.ac.uk)

---

## Requisitos del entorno

### Librerías Python

Las librerías empleadas y su versión:

| Librería | Varsión |
|---|---|
| `nibabel` | 5.3.3 |
| `nilearn` | 0.12.1 |
| `pandas` | 2.3.3 |
| `numpy` | 2.3.0 |
| `scikit-learn` | 1.8.0 |
| `optuna` | 4.8.0 |
| `seaborn` | 0.13.2 |
| `matplotlib` | 3.10.8 |

---

### Docker — Instalación general

Docker es necesario para ejecutar tanto HeudiConv (Notebook 1) como fMRIPrep (preprocesado). Si no está instalado:

- **Windows / macOS:** descargar e instalar [Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

### Docker — HeudiConv (Notebook 1)

HeudiConv se encarga de la conversión DICOM → BIDS. Para descargar la imagen:

```bash
docker pull nipy/heudiconv:1.3.4
```

Verificar que la imagen está disponible:

```bash
docker images | findstr heudiconv
```

---

### Docker — fMRIPrep (preprocesado de imágenes)

Las imágenes T1w se preprocesaron con **fMRIPrep** ejecutado sobre Docker en modalidad `--anat-only`, normalizando al espacio estándar **MNI152NLin2009cAsym**. Para descargar la imagen:

```bash
docker pull nipreps/fmriprep:25.2.3
```

Verificar que la imagen está disponible:

```bash
docker images | findstr fmriprep
```

El comando empleado para el preprocesado de cada grupo fue el siguiente (sustituir `XXXSXXXX` por los identificadores reales de los sujetos y ajustar las rutas locales):

```bash
docker run --rm -v "/ruta/datos/BIDS_dataset:/data:ro" 
-v "/ruta/salida/fmriprep_output:/out" 
-v "/ruta/license_freesurfer/license.txt:/opt/freesurfer/license.txt" 
nipreps/fmriprep:latest /data /out participant 
--participant-label sub-001 sub-002 sub-003 sub-004 sub-005 
--anat-only 
--fs-license-file /opt/freesurfer/license.txt 
--output-spaces MNI152NLin2009cAsym 
--nthreads 24 
--omp-nthreads 4 
--mem 26000
```

Los parámetros clave del comando son:

| Parámetro | Valor | Descripción |
|---|---|---|
| `--anat-only` | — | Solo preprocesa la imagen estructural T1w |
| `--output-spaces` | `MNI152NLin2009cAsym` | Espacio estándar de normalización |
| `--fs-license-file` | `/opt/freesurfer/license.txt` | Licencia de FreeSurfer (requerida) |
| `--nthreads` | `24` | Número de hilos de procesamiento |
| `--omp-nthreads` | `4` | Hilos OpenMP por proceso |
| `--mem` | `26000` | Memoria RAM disponible en MB |

> Se requiere una licencia válida de FreeSurfer. Puede obtenerse gratuitamente en [surfer.nmr.mgh.harvard.edu/registration.html](https://surfer.nmr.mgh.harvard.edu/registration.html)

---

## Datos y licencia ADNI

Los datos empleados en este estudio proceden de **ADNI (Alzheimer's Disease Neuroimaging Initiative)**, un consorcio público-privado financiado por el National Institute on Aging, el National Institute of Biomedical Imaging and Bioengineering y compañías farmacéuticas y empresas de diagnóstico por imagen.

El acceso a los datos requiere registro y aprobación en 🔗 [https://adni.loni.usc.edu](https://adni.loni.usc.edu)

Los términos completos del acuerdo de uso están disponibles en [`ADNI Data Use Agreement/`](./ADNI%20Data%20Use%20Agreement/ADNI_Data_Use_Agreement.pdf). Entre las restricciones principales:

- Prohibida la redistribución de imágenes de pacientes
- Prohibida la publicación de datasets completos
- Permitido el uso del código de análisis sin datos adjuntos
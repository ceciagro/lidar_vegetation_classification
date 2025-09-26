# 🌲 Vegetation Classification with LiDAR  
# 🌲 Clasificación de Vegetación con LiDAR  

This project demonstrates a **professional workflow** for processing and classifying LiDAR data, with the goal of identifying and quantifying vegetation (e.g., grass, citrus, palm trees) over a defined area.  
It includes **point cloud preprocessing with PDAL**, **feature extraction with Python**, and initial classification using both **rule-based methods and machine learning models**.  

Este proyecto demuestra un **flujo de trabajo profesional** para procesar y clasificar datos LiDAR, con el objetivo de identificar y cuantificar vegetación (ej. pasto, cítricos, palmeras) en un área definida.  
Incluye el **preprocesamiento de nubes de puntos con PDAL**, la **extracción de features con Python** y la clasificación inicial mediante **reglas heurísticas y modelos de machine learning**.  

---

## 🎯 Objectives / Objetivos
- ✅ Practice with real-world LiDAR datasets.  
- ✅ Build a reproducible pipeline using PDAL and Python.  
- ✅ Train and validate classification models (rule-based and ML).  
- ✅ Generate maps and statistics suitable for technical reports.  

---

## 📂 Dataset
- **Source / Fuente**: New Zealand National LiDAR Program (**LINZ – Land Information New Zealand**)  
- **Access / Acceso**: [https://data.linz.govt.nz](https://data.linz.govt.nz)  
- **License / Licencia**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  

> **Required attribution / Atribución obligatoria**:  
> “Data sourced from the New Zealand LiDAR program, Land Information New Zealand (LINZ), licensed under CC-BY 4.0.”

---

## ⚙️ Methodology / Metodología  

The project is structured in **eight Jupyter notebooks**, each one focusing on a specific step of the pipeline.  

El proyecto está estructurado en **nueve notebooks de Jupyter**, cada uno dedicado a un paso del pipeline.  

| Notebook | EN (English) | ES (Español) |
|----------|--------------|--------------|
| 00_prepare_raw.ipynb | Prepare a subset (`sample.laz`) from the original `.laz` tiles. | Preparar un subconjunto (`sample.laz`) a partir de los tiles `.laz` originales. |
| 01_pipelines_setup.ipynb | Save reproducible PDAL JSON pipelines (e.g., `pipeline_make_hag.json`, `pipeline_dbscan.json`). | Guardar los pipelines PDAL en formato JSON (ej. `pipeline_make_hag.json`, `pipeline_dbscan.json`). |
| 02_hag_processing.ipynb | Compute **Height Above Ground (HAG)** using ground filtering (PMF). | Calcular la **Altura sobre el Terreno (HAG)** aplicando filtrado de suelo (PMF). |
| 03_veg_filter.ipynb | Filter vegetation points with **HAG ≥ 2 m**. | Filtrar vegetación con **HAG ≥ 2 m**. |
| 04_dbscan.ipynb | Cluster vegetation using **DBSCAN** (tree crowns, shrubs). | Agrupar vegetación con **DBSCAN** (copas, arbustos). |
| 05_exploration.ipynb | Initial analysis – histograms, statistics, 2D/3D plots. | Exploración inicial – histogramas, estadísticas, gráficos 2D/3D. |
| 06_cluster_metrics_and_labels.ipynb | Compute metrics (heights, density, area) and apply **heuristic labels**: shrub, low tree, medium tree, tall tree. | Calcular métricas (alturas, densidad, área) y aplicar **etiquetas heurísticas**: arbusto, árbol bajo, árbol medio, árbol alto. |
| 06_end_to_end.ipynb | Run the **full pipeline in one go**: raw → HAG → ≥2m → DBSCAN → clean CSV/Parquet. | Ejecutar el **pipeline completo de una vez**: raw → HAG → ≥2m → DBSCAN → CSV/Parquet limpio. |
| 07_advanced_metrics_ml.ipynb | Extract advanced features (vertical histograms, roughness, 3D shape descriptors) and test ML models (PCA + KMeans, Random Forest). | Extraer features avanzados (histogramas verticales, rugosidad, descriptores de forma 3D) y probar modelos de ML (PCA + KMeans, Random Forest). |
| 08_model_selection.ipynb | Compare supervised models (Random Forest, XGBoost, SVM, KNN, Logistic Regression) with **GridSearchCV**; evaluate with confusion matrix & feature importance. | Comparar modelos supervisados (Random Forest, XGBoost, SVM, KNN, Regresión Logística) con **GridSearchCV**; evaluar con matriz de confusión e importancia de variables. |
---

## 📊 Outputs / Resultados
- `veg_gt2m_dbscan_clean.csv` → Clean clustered points (noise removed). / Puntos clusterizados limpios (ruido eliminado).  
- `veg_gt2m_dbscan_clean.parquet` → Same dataset in Parquet format (efficient for ML). / Mismo dataset en formato Parquet (eficiente para ML).  
- `veg_gt2m_labeled_points.csv` → Labeled points by cluster. / Puntos etiquetados por cluster.  
- `cluster_metrics.csv` → Metrics per cluster. / Métricas por cluster.  
- `cluster_features.csv` → Enriched tabular dataset with advanced features. / Dataset enriquecido con variables avanzadas.  
- Figures and plots → 2D/3D visualizations of vegetation clusters. / Visualizaciones 2D/3D de clusters de vegetación.  

![Cluster Example](figures/cluster_3d_top5.png)

---

## 🚀 Next Steps / Próximos pasos
- Replace bounding box area with **convex hull / alpha-shape** for better canopy estimates.  
- Integrate **field data** for supervised classification (species-level).  
- Scale the pipeline to **large datasets and cloud environments**.  

---

## 📜 Repository License / Licencia del Repositorio
- **Code / Código**: MIT License.  
- **Data / Datos**: LINZ – CC-BY 4.0.  

---

## 👩‍💻 Author / Autora
**Cecilia Ledesma** – Data Scientist & Machine Learning Engineer  

- 🌐 Portfolio: [Upwork Profile](https://www.upwork.com/freelancers/cledesma)  
- 📂 GitHub: [https://github.com/ceciagro]  






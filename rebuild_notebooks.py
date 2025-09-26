# rebuild_notebooks.py
import json, os
from pathlib import Path

try:
    import nbformat as nbf
except ImportError:
    raise SystemExit("Instala nbformat:  pip install nbformat")

ROOT = Path(".")
NB_DIR = ROOT / "notebooks"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
PIPES = ROOT / "pipelines"
FIGS = ROOT / "figures"

for d in [NB_DIR, DATA, RAW, PROC, PIPES, FIGS, NB_DIR/"data", NB_DIR/"data"/"raw", NB_DIR/"data"/"processed", NB_DIR/"figures", NB_DIR/"pipelines"]:
    d.mkdir(parents=True, exist_ok=True)

def nb(*cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = []
    for c in cells:
        if c["type"] == "md":
            nb["cells"].append(nbf.v4.new_markdown_cell(c["src"]))
        else:
            nb["cells"].append(nbf.v4.new_code_cell(c["src"]))
    return nb

def w(name, nb_obj):
    out = NB_DIR / name
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb_obj, f)
    print("✔ creado", out)

# 00_prepare_raw.ipynb
nb_00 = nb(
{"type":"md","src":"# 00 — Prepare raw sample\\n\\nCopia uno de tus `.laz` originales a `data/raw/sample.laz` y valida con `pdal info`. Ajusta `SAMPLE_SOURCE` si querés usar otro tile."},
{"type":"code","src":r'''
# === Celda 1: Rutas y utilidades ===
import os, shutil, subprocess, json
from pathlib import Path

# === Configurá acá tus archivos fuente (.laz) ===
SRC1 = Path("/Users/cecilialedesma/Library/Mobile Documents/com~apple~CloudDocs/projects_2025/lidar_vegetation_classification/raw/ot_CL1_WLG_2013_1km_059111.laz")
SRC2 = Path("/Users/cecilialedesma/Library/Mobile Documents/com~apple~CloudDocs/projects_2025/lidar_vegetation_classification/raw/ot_CL1_WLG_2013_1km_096068.laz")

# Elegí cuál usar como 'sample'
SAMPLE_SOURCE = SRC1

ROOT = Path(".").resolve()
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
PROC = DATA / "processed"
PIPES = ROOT / "pipelines"
for d in (RAW_DIR, PROC, PIPES): d.mkdir(parents=True, exist_ok=True)

SAMPLE = RAW_DIR / "sample.laz"

def run(cmd:list[str]) -> str:
    print("$", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout: print(p.stdout)
    if p.returncode != 0:
        if p.stderr: print(p.stderr)
        raise RuntimeError(f"Fallo: {' '.join(cmd)}")
    return p.stdout

print("SAMPLE_SOURCE:", SAMPLE_SOURCE)
print("SAMPLE destino:", SAMPLE)
'''},
{"type":"code","src":r'''
# === Celda 2: Copiar/actualizar sample.laz ===
assert SAMPLE_SOURCE.exists(), f"No existe el archivo fuente: {SAMPLE_SOURCE}"
shutil.copy2(SAMPLE_SOURCE, SAMPLE)
print(f"✅ Copiado: {SAMPLE_SOURCE} → {SAMPLE}")
'''},
{"type":"code","src":r'''
# === Celda 3: (Opcional) Validación rápida con PDAL ===
try:
    run(["pdal", "info", str(SAMPLE), "--summary"])
    print("✅ pdal info OK sobre sample.laz")
except FileNotFoundError:
    print("⚠️ PDAL no está en PATH. Saltando validación.")
'''},
)
w("00_prepare_raw.ipynb", nb_00)

# 01_pipelines_setup.ipynb
nb_01 = nb(
{"type":"md","src":"# 01 — Pipelines setup\\n\\nGenera pipelines PDAL reproducibles: HAG, filtro ≥2m y DBSCAN."},
{"type":"code","src":r'''
# === Celda 1: Rutas y parámetros ===
import json
from pathlib import Path

ROOT = Path(".").resolve()
PIPES = ROOT / "pipelines"
DATA  = ROOT / "data"
RAW   = DATA / "raw" / "sample.laz"
PROC  = DATA / "processed"

PIPES.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

DBSCAN_EPS = 0.8
DBSCAN_MINPTS = 20

LAS_HAG = PROC / "sample_hag.las"
LAS_VEG = PROC / "veg_gt2m.las"
LAS_DB  = PROC / "veg_gt2m_dbscan.las"
CSV_DB  = PROC / "veg_gt2m_dbscan.csv"
CSV_DB_CLEAN = PROC / "veg_gt2m_dbscan_clean.csv"

print("Pipelines →", PIPES)
print("RAW esperado →", RAW)
'''},
{"type":"code","src":r'''
# === Celda 2: pipeline_make_hag.json ===
pipeline_make_hag = {
  "pipeline": [
    {"type": "readers.las", "filename": str(RAW)},
    {"type": "filters.pmf",
     "max_window_size": 18, "slope": 0.15,
     "initial_distance": 0.5, "cell_size": 1.0},
    {"type": "filters.hag_delaunay"},
    {"type": "writers.las",
     "filename": str(LAS_HAG),
     "minor_version": 4, "dataformat_id": 6,
     "extra_dims": "all"}
  ]
}
( PIPES / "pipeline_make_hag.json").write_text(json.dumps(pipeline_make_hag, indent=2))
print("✅ Escrito:", PIPES / "pipeline_make_hag.json")
'''},
{"type":"code","src":r'''
# === Celda 3: pipeline_filter_veg.json (HAG ≥ 2 m) ===
pipeline_filter_veg = {
  "pipeline": [
    str(LAS_HAG),
    {"type": "filters.range", "limits": "HeightAboveGround[2:]"},
    {"type": "writers.las",
     "filename": str(LAS_VEG),
     "minor_version": 4, "dataformat_id": 6,
     "extra_dims": "all"}
  ]
}
( PIPES / "pipeline_filter_veg.json").write_text(json.dumps(pipeline_filter_veg, indent=2))
print("✅ Escrito:", PIPES / "pipeline_filter_veg.json")
'''},
{"type":"code","src":r'''
# === Celda 4: pipeline_dbscan.json (deja ruido) ===
pipeline_dbscan = {
  "pipeline": [
    str(LAS_VEG),
    {"type": "filters.dbscan",
     "min_points": DBSCAN_MINPTS,
     "eps": DBSCAN_EPS,
     "dimensions": "X,Y,Z"},
    {"type": "writers.las",
     "filename": str(LAS_DB),
     "minor_version": 4, "dataformat_id": 6,
     "extra_dims": "all"},
    {"type": "writers.text",
     "filename": str(CSV_DB),
     "format": "csv",
     "order": "X,Y,Z,HeightAboveGround,ClusterID",
     "keep_unspecified": False,
     "quote_header": False}
  ]
}
( PIPES / "pipeline_dbscan.json").write_text(json.dumps(pipeline_dbscan, indent=2))
print("✅ Escrito:", PIPES / "pipeline_dbscan.json")
'''},
{"type":"code","src":r'''
# === Celda 5: cluster_dbscan.json (quita ruido en el pipeline) ===
cluster_dbscan = {
  "pipeline": [
    str(LAS_VEG),
    {"type": "filters.dbscan",
     "min_points": DBSCAN_MINPTS,
     "eps": DBSCAN_EPS,
     "dimensions": "X,Y,Z"},
    {"type": "filters.range", "limits": "ClusterID![-1]"},
    {"type": "writers.las",
     "filename": str(PROC / "veg_gt2m_dbscan_clean.las"),
     "minor_version": 4, "dataformat_id": 6,
     "extra_dims": "all"},
    {"type": "writers.text",
     "filename": str(CSV_DB_CLEAN),
     "format": "csv",
     "order": "X,Y,Z,HeightAboveGround,ClusterID",
     "keep_unspecified": False,
     "quote_header": False}
  ]
}
( PIPES / "cluster_dbscan.json").write_text(json.dumps(cluster_dbscan, indent=2))
print("✅ Escrito:", PIPES / "cluster_dbscan.json")
'''},
{"type":"code","src":r'''
print("""
Ejecutá en orden (terminal):
1) pdal pipeline pipelines/pipeline_make_hag.json
2) pdal pipeline pipelines/pipeline_filter_veg.json
3) pdal pipeline pipelines/pipeline_dbscan.json   (o)   pdal pipeline pipelines/cluster_dbscan.json
""")
'''},
)
w("01_pipelines_setup.ipynb", nb_01)

# 02_hag_processing.ipynb
nb_02 = nb(
{"type":"md","src":"# 02 — HAG processing\\n\\nCorre el pipeline de HAG y valida que tenga `HeightAboveGround`."},
{"type":"code","src":r'''
import subprocess, json
from pathlib import Path

ROOT = Path(".").resolve()
DATA = ROOT / "data"
RAW  = DATA / "raw" / "sample.laz"
PROC = DATA / "processed"
PIPES = ROOT / "pipelines"
LAS_HAG = PROC / "sample_hag.las"
PIPE_MAKE_HAG = PIPES / "pipeline_make_hag.json"

for d in (PROC, PIPES): d.mkdir(parents=True, exist_ok=True)

assert RAW.exists(), f"Falta RAW: {RAW}. Corré 00_prepare_raw.ipynb"
assert PIPE_MAKE_HAG.exists(), f"Falta pipeline: {PIPE_MAKE_HAG}. Corré 01_pipelines_setup.ipynb"

def run(cmd):
    print("$", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout: print(p.stdout)
    if p.returncode != 0:
        print(p.stderr); raise RuntimeError("Fallo de comando")
    return p.stdout

run(["pdal", "pipeline", str(PIPE_MAKE_HAG)])
out = run(["pdal", "info", str(LAS_HAG), "--all"])
print("¿Incluye HeightAboveGround?", "HeightAboveGround" in out)
'''},
)
w("02_hag_processing.ipynb", nb_02)

# 03_veg_filter.ipynb
nb_03 = nb(
{"type":"md","src":"# 03 — Vegetation filter (HAG ≥ 2 m)"},
{"type":"code","src":r'''
import subprocess, json
from pathlib import Path

ROOT = Path(".").resolve()
DATA = ROOT / "data"
PROC = DATA / "processed"
PIPES = ROOT / "pipelines"

LAS_HAG = PROC / "sample_hag.las"
LAS_VEG = PROC / "veg_gt2m.las"
PIPE_FILTER_VEG = PIPES / "pipeline_filter_veg.json"

assert LAS_HAG.exists(), f"Falta HAG: {LAS_HAG}. Corré 02_hag_processing.ipynb"
assert PIPE_FILTER_VEG.exists(), f"Falta pipeline: {PIPE_FILTER_VEG}. Corré 01_pipelines_setup.ipynb"

def run(cmd):
    print("$", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout: print(p.stdout)
    if p.returncode != 0:
        print(p.stderr); raise RuntimeError("Fallo")
    return p.stdout

run(["pdal", "pipeline", str(PIPE_FILTER_VEG)])

all_info = run(["pdal", "info", str(LAS_VEG), "--all"])
print("¿veg_gt2m.las incluye HeightAboveGround?", "HeightAboveGround" in all_info)

summary = run(["pdal", "info", str(LAS_VEG), "--summary"])
meta = json.loads(summary)["summary"]["metadata"]
print(f"Formato LAS 1.{meta['minor_version']} / dataformat_id={meta['dataformat_id']}")
'''},
)
w("03_veg_filter.ipynb", nb_03)

# 04_dbscan.ipynb
nb_04 = nb(
{"type":"md","src":"# 04 — DBSCAN clustering\\n\\nCorre DBSCAN (CSV+LAS) y muestra head del CSV."},
{"type":"code","src":r'''
import subprocess
from pathlib import Path

ROOT = Path(".").resolve()
DATA = ROOT / "data"
PROC = DATA / "processed"
PIPES = ROOT / "pipelines"

LAS_VEG = PROC / "veg_gt2m.las"
LAS_DB  = PROC / "veg_gt2m_dbscan.las"
CSV_DB  = PROC / "veg_gt2m_dbscan.csv"

PIPE_DBSCAN        = PIPES / "pipeline_dbscan.json"
PIPE_CLUSTER_CLEAN = PIPES / "cluster_dbscan.json"

assert LAS_VEG.exists(), f"Falta veg_gt2m.las: {LAS_VEG}. Corré 03_veg_filter.ipynb"
assert PIPE_DBSCAN.exists(), "Falta pipeline_dbscan.json (01_pipelines_setup.ipynb)"

def run(cmd):
    print("$", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout: print(p.stdout)
    if p.returncode != 0:
        print(p.stderr); raise RuntimeError("Fallo")
    return p.stdout

# Opción A (deja ruido):
run(["pdal", "pipeline", str(PIPE_DBSCAN)])

# Chequeos
if LAS_DB.exists():
    run(["pdal", "info", str(LAS_DB), "--summary"])
else:
    print("Aviso: no se encontró", LAS_DB)

if CSV_DB.exists():
    try:
        run(["bash", "-lc", f"head -n 5 '{CSV_DB}'"])
    except Exception:
        print(open(CSV_DB).read().splitlines()[:5])
else:
    print("Aviso: no se encontró", CSV_DB)
'''},
)
w("04_dbscan.ipynb", nb_04)

# 05_exploration.ipynb
nb_05 = nb(
{"type":"md","src":"# 05 — Exploration & plots\\n\\nLimpia ruido (ClusterID=-1), genera tablas y gráficos (hist/boxplot/XY/3D)."},
{"type":"code","src":r'''
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import plotly.express as px

ROOT = Path(".").resolve()
PROC = ROOT / "data" / "processed"

CSV_DBSCAN   = PROC / "veg_gt2m_dbscan.csv"
CSV_CLEAN    = PROC / "veg_gt2m_dbscan_clean.csv"
PARQ_CLEAN   = PROC / "veg_gt2m_dbscan_clean.parquet"

assert CSV_DBSCAN.exists(), f"Falta CSV DBSCAN: {CSV_DBSCAN}. Corré 04_dbscan.ipynb"

usecols = ["X","Y","Z","HeightAboveGround","ClusterID"]
dtypes  = {"X":"float32","Y":"float32","Z":"float32","HeightAboveGround":"float32","ClusterID":"int32"}
df = pd.read_csv(CSV_DBSCAN, usecols=usecols, dtype=dtypes)

df_clean = df[df["ClusterID"] != -1].copy()
df_clean.to_csv(CSV_CLEAN, index=False)
df_clean.to_parquet(PARQ_CLEAN, index=False)

print(f"Puntos originales: {len(df):,} | sin ruido: {len(df_clean):,}")
print("CSV clean →", CSV_CLEAN)
print("Parquet clean →", PARQ_CLEAN)

cluster_counts = (df_clean["ClusterID"].value_counts()
                  .rename_axis("ClusterID").reset_index(name="#puntos"))
stats = (df_clean.groupby("ClusterID")["HeightAboveGround"]
         .agg(n="count", mean="mean",
              p25=lambda s: s.quantile(0.25),
              median="median",
              p75=lambda s: s.quantile(0.75),
              max="max").reset_index().sort_values("n", ascending=False))

cluster_counts.to_csv(PROC / "cluster_sizes.csv", index=False)
stats.to_csv(PROC / "cluster_hag_stats.csv", index=False)

# Hist tamaños
plt.figure(figsize=(6,4))
plt.hist(cluster_counts["#puntos"], bins=50)
plt.xlabel("Tamaño del cluster (# puntos)")
plt.ylabel("Frecuencia")
plt.title("Distribución de tamaños de clusters (DBSCAN)")
plt.tight_layout(); plt.show()

# Boxplot top10
top_ids = cluster_counts["ClusterID"].head(10).tolist()
subset = df_clean[df_clean["ClusterID"].isin(top_ids)]
plt.figure(figsize=(10,5))
subset.boxplot(column="HeightAboveGround", by="ClusterID", grid=False, rot=45)
plt.ylabel("HeightAboveGround (m)")
plt.title("Alturas por cluster (Top 10)")
plt.suptitle(""); plt.tight_layout(); plt.show()

# XY muestra
sample = df_clean.sample(min(50_000, len(df_clean)), random_state=42)
plt.figure(figsize=(6,6))
plt.scatter(sample["X"], sample["Y"], c=sample["ClusterID"], s=1, cmap="tab20", alpha=0.6)
plt.xlabel("X"); plt.ylabel("Y"); plt.title("Vista XY por ClusterID (muestra)")
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout(); plt.show()

# 3D top-5
dfc = df_clean[["X","Y","HeightAboveGround","ClusterID"]].copy()
top5 = dfc["ClusterID"].value_counts().head(5).index.tolist()
sub = dfc[dfc["ClusterID"].isin(top5)].copy()
if len(sub) > 20_000: sub = sub.sample(20_000, random_state=42)
sub["Xn"] = sub["X"] - sub["X"].min()
sub["Yn"] = sub["Y"] - sub["Y"].min()
sub["ClusterID_str"] = sub["ClusterID"].astype(int).astype(str)

fig = px.scatter_3d(sub, x="Xn", y="Yn", z="HeightAboveGround",
                    color="ClusterID_str",
                    title="Clusters en 3D (submuestra normalizada)")
fig.update_traces(marker=dict(size=2, opacity=0.5))
fig.update_layout(scene_aspectmode="data"); fig.show()
'''},
)
w("05_exploration.ipynb", nb_05)

# 06_cluster_metrics_and_labels.ipynb
nb_06a = nb(
{"type":"md","src":"# 06 — Cluster metrics & heuristic labels\\n\\nCalcula métricas por cluster y etiqueta por p95 de altura."},
{"type":"code","src":r'''
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

LABEL_RULES = [
    ("arbusto",      0,   5),
    ("árbol_bajo",   5,  12),
    ("árbol_medio", 12,  25),
    ("árbol_alto",  25, 1e9),
]

ROOT = Path("."); DATA = ROOT/"data"; PROC = DATA/"processed"
IN_POINTS = PROC/"veg_gt2m_dbscan_clean.csv"
OUT_CLUSTER_METRICS = PROC/"cluster_metrics.csv"
OUT_POINTS_LABELED  = PROC/"veg_gt2m_labeled_points.csv"
assert IN_POINTS.exists(), f"No encuentro {IN_POINTS}. Corré 05_exploration.ipynb antes."

usecols = ["X","Y","HeightAboveGround","ClusterID"]
dtypes  = {"X":"float32","Y":"float32","HeightAboveGround":"float32","ClusterID":"int32"}
df = pd.read_csv(IN_POINTS, usecols=usecols, dtype=dtypes)
print(f"Puntos: {len(df):,} | Clusters: {df['ClusterID'].nunique():,}")

# Métricas por cluster
agg = df.groupby("ClusterID")["HeightAboveGround"].agg(
    n="count", h_min="min",
    h_q25=lambda s: s.quantile(0.25),
    h_med="median", h_mean="mean",
    h_q75=lambda s: s.quantile(0.75),
    h_p95=lambda s: s.quantile(0.95),
    h_max="max").reset_index()

xy = df.groupby("ClusterID").agg(
    x_min=("X","min"), x_max=("X","max"),
    y_min=("Y","min"), y_max=("Y","max"),
    cx=("X","mean"),  cy=("Y","mean")).reset_index()

xy["dx"]=(xy["x_max"]-xy["x_min"]).astype("float32")
xy["dy"]=(xy["y_max"]-xy["y_min"]).astype("float32")
xy["area_bbox_m2"]=(xy["dx"]*xy["dy"]).astype("float32")

metrics = agg.merge(xy, on="ClusterID", how="left")
metrics["pt_density"] = metrics["n"]/metrics["area_bbox_m2"].replace({0:np.nan})
metrics["pt_density"] = metrics["pt_density"].fillna(0).astype("float32")
metrics = metrics.sort_values("n", ascending=False).reset_index(drop=True)

def assign_label(h: float, rules=LABEL_RULES) -> str:
    for name, lo, hi in rules:
        if lo <= h < hi: return name
    return "sin_label"

metrics["label_v1"] = metrics["h_p95"].apply(assign_label)
DENSE_THRESH = 0.08
mask = (metrics["label_v1"]=="arbusto") & (metrics["pt_density"]>DENSE_THRESH)
metrics.loc[mask, "label_v1"] = "matorral/arbusto"
metrics["label"] = metrics["label_v1"]

# Export
metrics.to_csv(OUT_CLUSTER_METRICS, index=False)
df_labels = metrics[["ClusterID","label"]]
df_labeled = df.merge(df_labels, on="ClusterID", how="left")
df_labeled.to_csv(OUT_POINTS_LABELED, index=False)
print("✔ Guardado:", OUT_CLUSTER_METRICS, "y", OUT_POINTS_LABELED)

# Gráficos rápidos
plt.figure(); plt.hist(metrics["h_p95"], bins=40)
plt.xlabel("Altura p95 (m)"); plt.ylabel("Frecuencia"); plt.title("p95 por cluster"); plt.tight_layout(); plt.show()
'''},
)
w("06_cluster_metrics_and_labels.ipynb", nb_06a)

# 06_end_to_end.ipynb
nb_06e = nb(
{"type":"md","src":"# 06 — End-to-End (raw → HAG → ≥2m → DBSCAN → clean)\\n\\nCorre todo el pipeline de punta a punta y valida salidas."},
{"type":"code","src":r'''
# 1) Config & helper
import os, json, subprocess, shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RAW_SRC_DIR = Path("./raw")
SAMPLE_NAME = "sample.laz"
DBSCAN_EPS = 0.8
DBSCAN_MINPTS = 20

ROOT = Path(".").resolve()
DATA = ROOT/"data"
RAW_DST_DIR = DATA/"raw"
PROC = DATA/"processed"
PIPES = ROOT/"pipelines"

RAW = RAW_DST_DIR / SAMPLE_NAME
LAS_HAG = PROC/"sample_hag.las"
LAS_VEG = PROC/"veg_gt2m.las"
LAS_DB  = PROC/"veg_gt2m_dbscan.las"
CSV_DBSCAN = PROC/"veg_gt2m_dbscan.csv"
CSV_CLEAN  = PROC/"veg_gt2m_dbscan_clean.csv"
PARQ_CLEAN = PROC/"veg_gt2m_dbscan_clean.parquet"

PIPE_HAG    = PIPES/"pipeline_make_hag.json"
PIPE_DBSCAN = PIPES/"pipeline_dbscan.json"

for d in [RAW_DST_DIR, PROC, PIPES]: d.mkdir(parents=True, exist_ok=True)

def run(cmd):
    print("$", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout: print(p.stdout)
    if p.returncode != 0:
        if p.stderr: print(p.stderr)
        raise RuntimeError(f"Fallo: {' '.join(cmd)}")
    return p.stdout

print("=== Config ==="); print("ROOT:", ROOT); print("RAW destino:", RAW)

# 2) Preparar sample.laz
if not RAW.exists():
    candidates = sorted(RAW_SRC_DIR.glob("*.laz"))
    if not candidates:
        raise FileNotFoundError("Poné originales en ./raw o cambia RAW_SRC_DIR")
    src = candidates[0]; shutil.copy2(src, RAW); print("✅ Copiado", src, "→", RAW)
else:
    print("Usando RAW existente:", RAW)

run(["pdal","info",str(RAW),"--summary"])

# 3) Crear/actualizar pipelines
pipeline_hag = {
    "pipeline":[
        {"type":"readers.las","filename":str(RAW)},
        {"type":"filters.pmf","max_window_size":18,"slope":0.15,"initial_distance":0.5,"cell_size":1.0},
        {"type":"filters.hag_delaunay"},
        {"type":"writers.las","filename":str(LAS_HAG),"minor_version":4,"dataformat_id":6,"extra_dims":"all"}
    ]
}
PIPE_HAG.write_text(json.dumps(pipeline_hag, indent=2))

pipeline_dbscan = {
  "pipeline":[
    str(LAS_VEG),
    {"type":"filters.dbscan","min_points":DBSCAN_MINPTS,"eps":DBSCAN_EPS,"dimensions":"X,Y,Z"},
    {"type":"writers.las","filename":str(LAS_DB),"minor_version":4,"dataformat_id":6,"extra_dims":"all"},
    {"type":"writers.text","filename":str(CSV_DBSCAN),"format":"csv","order":"X,Y,Z,HeightAboveGround,ClusterID","keep_unspecified":False,"quote_header":False}
  ]
}
PIPE_DBSCAN.write_text(json.dumps(pipeline_dbscan, indent=2))

# 4) HAG
run(["pdal","pipeline",str(PIPE_HAG)])
hag_all = run(["pdal","info",str(LAS_HAG),"--all"])
print("¿HeightAboveGround?", "HeightAboveGround" in hag_all)

# 5) ≥2m
run(["pdal","translate",str(LAS_HAG),str(LAS_VEG),"range",
     "--filters.range.limits=HeightAboveGround[2:]",
     "--writers.las.minor_version=4",
     "--writers.las.dataformat_id=6",
     "--writers.las.extra_dims=all"])
veg_all = run(["pdal","info",str(LAS_VEG),"--all"])
print("¿veg_gt2m con HAG?", "HeightAboveGround" in veg_all)

# 6) DBSCAN
run(["pdal","pipeline",str(PIPE_DBSCAN)])

# 7) Limpieza pandas
usecols=["X","Y","Z","HeightAboveGround","ClusterID"]
dtypes={"X":"float32","Y":"float32","Z":"float32","HeightAboveGround":"float32","ClusterID":"int32"}
df = pd.read_csv(CSV_DBSCAN, usecols=usecols, dtype=dtypes)
df_clean = df[df["ClusterID"] != -1].copy()
df_clean.to_csv(CSV_CLEAN, index=False)
df_clean.to_parquet(PARQ_CLEAN, index=False)

print(f"Puntos originales: {len(df):,} | sin ruido: {len(df_clean):,}")
print("CSV clean →", CSV_CLEAN, "| Parquet →", PARQ_CLEAN)

# 8) Hist rápido
cluster_counts = (df_clean["ClusterID"].value_counts()
                  .rename_axis("ClusterID").reset_index(name="#puntos"))
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.hist(cluster_counts["#puntos"], bins=50); plt.xlabel("Tamaño cluster"); plt.ylabel("Frecuencia"); plt.title("DBSCAN – tamaños")
plt.tight_layout(); plt.show()

print("✅ End-to-end completo.")
'''},
)
w("06_end_to_end.ipynb", nb_06e)

print("\nListo. Revisa la carpeta 'notebooks/'.")
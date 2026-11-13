# Object Detection in Gigapixel Images with YOLO (NJU-HARD) + RAG Analysis Agent

Pilot project to detect **pedestrians, cars, buses and trucks** in the very high-resolution frames (~100–120 MP) of the [`RL-MIND/NJU-HARD-Detection`](https://huggingface.co/datasets/RL-MIND/NJU-HARD-Detection) dataset using YOLO11n (`ultralytics`), and to **analyze the detections with an LLM agent** that answers natural-language questions through Retrieval-Augmented Generation (RAG).

Since the frames are too large to feed whole into the model, the pipeline **splits them into overlapping tiles** (1024 px, 128 px overlap), builds a dataset in YOLO format, trains the model, validates it, and lets you visualize its predictions. The agent then turns those predictions into structured data, indexes them in a vector database and lets you chat with them.

```
NJU-HARD (Hugging Face)
        │
        ▼
   yolo.py ──────────► runs_nju/*/weights/best.pt          (GPU: Colab / Linux + CUDA)
                              │
                              ▼
   agente.py exportar ──► resultados/detecciones/<frame>.json   (+ metricas.json)
                              │
                              ▼
   agente.py indexar ───► ChromaDB (data/chroma)               (local multilingual embeddings)
                              │
                              ▼
   agente.py chat ──────► LangGraph agent ──► buscar_resultados (RAG search)
                              │          └──► generar_grafica ──► resultados/graficas/*.png
                              └──► configurable LLM (Anthropic | OpenAI | Gemini | Ollama)
```

## Repository structure

| File | What it does |
|---|---|
| `yolo.py` | Detection pipeline: sequence scanning, parallel tiling, training, validation and a sample prediction. |
| `prediccion.py` | Draws the boxes predicted by the trained model, with confidence in %, on a validation image. |
| `agente.py` | RAG analysis agent in a single file: exports detections to JSON, indexes them in ChromaDB and answers questions with an LLM (CLI commands `exportar`, `indexar`, `chat`). |
| `tests/test_agente.py` | Agent test suite (`pytest`); runs without network access or API keys. |
| `requirements.txt` | Pinned dependencies (verified with Python 3.12 on macOS arm64). |
| `.env.example` | Template for all agent settings (LLM provider, keys, embeddings, paths). Copy it to `.env`. |
| `metricas.json` | Frame-level evaluation of the detector (after cross-tile deduplication), used as a knowledge source by the agent. |
| `GUIA_AGENTE.md` / `PROMPT_AGENTE.md` | Step-by-step guide and full specification of the agent (in Spanish). |

## Requirements

**Training (`yolo.py`, `prediccion.py`)**

- Linux (or Google Colab) with an **NVIDIA GPU and CUDA**; both scripts use `device=0` and `yolo.py` raises an error if no GPU is detected.
- Python 3.10+ and: `pip install torch ultralytics datasets pillow pyyaml tqdm`
- Hugging Face access to stream the dataset.
- Enough RAM: each decoded frame takes ~350 MB in RGB and several are processed in parallel. If you run out of memory, lower `NUM_WORKERS` in `yolo.py`.

**Analysis agent (`agente.py`)**

- Runs on a regular laptop: no NVIDIA GPU needed (it uses Apple Silicon `mps` automatically, or the CPU).
- **Python 3.12** recommended (some dependencies such as `torch` and `chromadb` are not yet stable on 3.14).
- An API key for one LLM provider (Anthropic, OpenAI or Google AI Studio), or a local [Ollama](https://ollama.com) installation.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
# or: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
cp .env.example .env        # then fill in your provider and API key
```

## Usage

### 1. Build the dataset, train and validate

```bash
python yolo.py
```

It runs four steps in order:

1. **Dataset**: streams the frames and generates the labeled tiles in `nju_yolo/` (with its `data.yaml`).
2. **Training**: trains `yolo11n` and saves the results in `runs_nju/`.
3. **Validation**: validates the best model (`best.pt`) and prints precision, recall, mAP50 and mAP50-95.
4. **Sample prediction**: predicts one validation image and saves the annotated result in `runs/detect/predict`.

It also works in Google Colab: if detected, it mounts Google Drive (you can change `BASE_DIR` to `Path("/content")`). If you train in Colab and analyze locally, copy `runs_nju/<run>/weights/best.pt` (and `nju_yolo/images/val/` if you want to analyze the validation tiles) to your machine.

### 2. Visualize predictions

Once the model is trained, without retraining:

```bash
python prediccion.py                  # picks a validation image with many detections
python prediccion.py path/image.jpg   # or uses the image you specify
```

It uses the most recent `best.pt` in `runs_nju/` and saves the result to `prediccion_ejemplo.jpg`, with each box labeled as `class NN%`. If none of the tested images has detections with confidence ≥ 0.25, it suggests lowering `CONF` (for example to 0.10).

### 3. Analyze the results with the LLM agent

```bash
source .venv/bin/activate
python agente.py exportar --fuente nju_yolo/images/val   # 1. run YOLO and save one JSON per frame
python agente.py indexar                                  # 2. build the RAG index
python agente.py chat                                     # 3. ask questions
```

See [RAG analysis agent](#rag-analysis-agent) below for details, chat commands and example questions.

## How the dataset is built

- **Sequence selection**: 5 sequences are used for training and 1 for validation. With `SEQUENCE_SELECTION = "first"`, the sequences that end earliest in the stream are chosen so the download stops as soon as possible; with `"random"`, they are chosen at random using `SEED`.
- **Tiling**: each frame is cut into 1024 px tiles with 128 px overlap and saved as JPEG (quality 90).
- **Labels**: converted to YOLO format (coordinates normalized per tile). Only objects that fall **entirely** inside a tile are kept; objects cut by the tile edge are omitted, and tiles with no objects are not saved.
- **Resumable and network-fault tolerant**: progress is saved in `nju_yolo/` (`done_*.json`, `failed_*.json`) and, if the connection drops, the stream is reopened up to 5 times. The sequence scan is cached in `nju_yolo/seqs_v2.json`.

## Main configuration

| Parameter | File | Default |
|---|---|---|
| `SEED` | `yolo.py` | `42` |
| `TRAIN_SEQUENCES` / `VAL_SEQUENCES` | `yolo.py` | `5` / `1` |
| `SEQUENCE_SELECTION` | `yolo.py` | `"first"` (or `"random"`) |
| `TILE_SIZE` / `OVERLAP` | `yolo.py` | `1024` / `128` |
| `NUM_WORKERS` | `yolo.py` | `min(4, CPUs - 1)` |
| `epochs` / `imgsz` / `batch` | `train_model()` in `yolo.py` | `2` / `640` / `2` |
| `CONF` / `IMGSZ` | `prediccion.py` | `0.25` / `640` |
| Agent settings (LLM, embeddings, RAG, paths) | `.env` | see [Configuration](#configuration-env) |

Classes: `0: pedestrian`, `1: car`, `2: bus`, `3: truck`.

> Training is configured as a pilot (2 epochs, batch size 2). For a real training run, increase `epochs` and `batch` in `train_model()`.

## RAG analysis agent

`agente.py` adds an LLM layer on top of the detector. Instead of reading JSON files or plots by hand, you ask questions in natural language and the agent answers with figures taken from the actual results, **citing the source of every number** and generating charts on demand.

### What it does

- **Answers free-form questions** about detections (totals, per-frame counts, spatial distribution, confidence) and about model quality (precision, recall, F1, TP/FP/FN per class and IoU threshold).
- **Cites its sources** in brackets, e.g. `[resumen_global]`, `[resumen_frame/UAV-BASE-13_6_00000088]`, `[metricas_modelo/IoU_0.5/car]`, so every figure can be traced back.
- **Generates charts** (bar charts, confidence histograms, spatial heatmaps, per-class metrics) as PNG files with exact values.
- **Refuses to invent data**: if the indexed results do not contain the answer, it says so instead of guessing.
- **Warns about caveats**: low precision/recall (counts may over- or under-estimate real objects) and partial coverage when frames were rebuilt from tiles.
- **Keeps conversation memory** within a session, so follow-up questions such as "and in which frame are there more?" work.
- **Answers in Spanish** (its system prompt is in Spanish, matching the course; questions can be asked in Spanish or English).

### How it works

1. **Export** (`agente.py exportar`): runs `best.pt` and writes one JSON per frame to `resultados/detecciones/`, with every detection (class, confidence, box in frame coordinates) plus a precomputed summary: counts per class, mean confidence per class and counts per cell of a `4x4` grid over the frame.
   - **Full frames**: the frame is cut into overlapping tiles, each tile is predicted, boxes are shifted to frame coordinates and duplicates in the overlaps are removed with class-wise NMS (`DEDUP_IOU`).
   - **Tiles from `nju_yolo/`** (`{frame_id}_{row}_{col}.jpg`): tiles are regrouped by frame. Since `yolo.py` only keeps tiles that contain labels and does not store the frame size, coverage is **partial** and positions are **approximate**; this is recorded in the JSON and the agent mentions it.
   - The mode is detected automatically (`--modo auto|frames|tiles`), and the device is chosen as `cuda` → `mps` → `cpu`.
2. **Index** (`agente.py indexar`): converts the JSON files and `metricas.json` into text documents with all figures **precomputed** (the LLM never counts boxes itself), embeds them with a local multilingual `sentence-transformers` model and stores them in ChromaDB. Re-indexing is idempotent (deterministic IDs, upsert) and removes documents of frames that no longer exist.
3. **Chat** (`agente.py chat`): a LangGraph agent (`langchain.agents.create_agent`) with two tools:
   - `buscar_resultados(consulta, tipo, frame_id)`: semantic search in ChromaDB, optionally filtered by document type and frame.
   - `generar_grafica(tipo_grafica, frame_id, clase, umbral_iou)`: reads the JSON files / `metricas.json` directly (not the RAG index), so chart values are exact.

Indexed document types:

| Type | Content | Typical question |
|---|---|---|
| `resumen_global` | Totals across all frames, class distribution, frames with most detections, vehicles. | "How many pedestrians in total?" |
| `resumen_frame` | Counts and mean confidence per class for one frame, its 3 busiest grid cells, inference parameters. | "How many detections in frame X?" |
| `region_frame` | Counts per grid cell of a frame (`r0c0` = top-left, `r3c3` = bottom-right). | "Where are pedestrians concentrated?" |
| `metricas_modelo` | `metricas.json` per IoU threshold and per class, with an interpretation of FP/FN. | "Why is car precision 0?" |

### Running it

```bash
source .venv/bin/activate

# 1. Export detections (uses the newest runs_nju/*/weights/best.pt, or YOLO_WEIGHTS in .env)
python agente.py exportar --fuente nju_yolo/images/val            # validation tiles
python agente.py exportar --fuente path/to/full_frame.jpg          # a full frame (complete coverage)
python agente.py exportar --fuente folder --pesos other_model.pt   # another model

# 2. Build the index (add --reset to rebuild from scratch)
python agente.py indexar

# 3a. Interactive chat (keeps conversation memory)
python agente.py chat

# 3b. Single question (useful for scripts or screenshots)
python agente.py chat -p "¿Qué clase tiene peor desempeño y por qué?"

# Tests (no network or API key needed)
pytest tests
```

Example session:

```
$ python agente.py chat
Cargando agente (google_genai / gemini-3.5-flash, effort=low)...
599 documentos indexados. Escribe tu pregunta o /ayuda.

tú> ¿Cuántos peatones se detectaron en total?
  · buscar_resultados(consulta="total peatones", tipo="resumen_global")

Se detectaron un total de 1228 peatones [resumen_global] en 123 frames...

tú> /fuentes
- [resumen_global] (distancia 0.xx)
  Resumen global de 123 frames analizados: 1228 detecciones en total...
```

Lines starting with `·` show which tool the agent called and with which arguments.

### Chat commands

| Command | What it does |
|---|---|
| `/fuentes` | Shows the documents retrieved for the last answer (to verify where each figure came from). |
| `/nuevo` | Starts a new conversation (forgets the previous context). |
| `/reindexar` | Re-indexes the results without leaving the chat (after a new export). |
| `/modelo` | Shows the active LLM provider and model. |
| `/ayuda` | Lists the commands. |
| `/salir` | Exits (also `Ctrl+D`). |

### Example use cases

| Use case | Example question (Spanish, as used in the chat) | What the agent does |
|---|---|---|
| Global totals | "¿Cuántos peatones se detectaron en total?" *(How many pedestrians were detected in total?)* | Searches `resumen_global` and cites it. |
| Busiest frame | "¿Qué frame tiene más detecciones y cuántas?" *(Which frame has the most detections?)* | Reads the ranking in `resumen_global`. |
| Spatial analysis | "¿En qué zona del frame UAV-BASE-13_6_00000088 se concentran los peatones?" *(Where are pedestrians concentrated?)* | Uses `resumen_frame` and `region_frame`; answers with grid cells (e.g. `r3c3`, bottom-right). |
| Model quality | "¿Por qué la precisión de la clase car es 0 si hubo detecciones?" *(Why is car precision 0?)* | Reads `metricas_modelo`: all 9 car predictions were false positives and recall is undefined. |
| Reliability of counts | "¿Es confiable el conteo de peatones según las métricas?" *(Is the pedestrian count reliable?)* | Combines counts with precision/recall and warns about over/under-estimation. |
| Coverage caveats | "¿Los conteos del frame UAV-BASE-13_6_00000088 representan todo el frame?" | Explains that tile-based exports have partial coverage. |
| Charts | "Hazme un heatmap de los peatones del frame UAV-BASE-13_6_00000088" / "Grafica el histograma de confianza de los peatones" / "Muéstrame las métricas por clase" | Calls `generar_grafica` and returns the PNG path plus the plotted values. |
| Out-of-scope questions | "¿Cuántas bicicletas hay en el frame UAV-BASE-13_6_00000088?" *(How many bicycles?)* | Says that bicycles are not one of the model's classes instead of inventing a number. |

Tips: write the full `frame_id` when asking about a specific frame, and use `/fuentes` whenever you want to check a figure.

### When to re-run each step

| If you change… | Re-run |
|---|---|
| The model (`best.pt`) or the images | `exportar` → `indexar` |
| `metricas.json` | `indexar` |
| `EMBEDDINGS_MODEL` | `indexar --reset` |
| The LLM provider/model in `.env` | Nothing; just reopen `chat` |

### Configuration (`.env`)

All agent settings live in `.env` (template: `.env.example`). `.env` is git-ignored, so API keys are never committed.

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER`, `LLM_MODEL` | Provider (`anthropic`, `openai`, `gemini`, `ollama`) and model name. Default: `anthropic` / `claude-opus-5-5`. Example for Gemini: `gemini` / `gemini-3.5-flash`. |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `OLLAMA_BASE_URL` | Credentials of the chosen provider (`GOOGLE_API_KEY` is a Google AI Studio key). Only the active provider's key is required. |
| `LLM_EFFORT` | Reasoning depth for Claude and Gemini (`low`…`max`; Gemini maps `xhigh`/`max` to `high`). Lower is faster. |
| `LLM_MAX_TOKENS` | Maximum output tokens. |
| `LLM_TEMPERATURE` | Only sent to OpenAI and Ollama (current Claude models reject it; Gemini keeps Google's recommended default). |
| `LLM_TIMEOUT`, `LLM_MAX_RETRIES` | Per-call timeout (s) and retries, so a provider outage fails fast instead of hanging the chat. |
| `EMBEDDINGS_MODEL`, `EMBEDDINGS_DEVICE` | Local `sentence-transformers` model and device (`mps` / `cpu` / `cuda`). |
| `CHROMA_DIR`, `CHROMA_COLLECTION`, `RAG_TOP_K` | Vector database location, collection name and number of retrieved chunks. |
| `YOLO_WEIGHTS`, `YOLO_CONF`, `YOLO_IMGSZ`, `TILE_SIZE`, `TILE_OVERLAP`, `DEDUP_IOU` | Inference and tiling parameters used by `exportar` (empty `YOLO_WEIGHTS` = newest `best.pt` in `runs_nju/`). |
| `GRID_FILAS`, `GRID_COLUMNAS` | Grid used to describe where detections are located (default `4x4`). |
| `DETECCIONES_DIR`, `METRICAS_PATH`, `GRAFICAS_DIR` | Input/output paths. |
| `LANGSMITH_*` | Optional tracing of agent runs in LangSmith. |

### Troubleshooting

| Message | Fix |
|---|---|
| `Falta GOOGLE_API_KEY en .env` (or another key) | Add the key of the active provider, or change `LLM_PROVIDER`. |
| `503 UNAVAILABLE` / "servicio saturado" | The provider is overloaded: retry later, lower `LLM_EFFORT` or switch `LLM_MODEL`. |
| `429 RESOURCE_EXHAUSTED` | Quota exhausted (e.g. Google AI Studio free tier): wait for the reset, enable billing or switch provider. |
| `No hay best.pt en runs_nju` | Copy the trained model there or set `YOLO_WEIGHTS`. |
| "La base vectorial está vacía" | Run `python agente.py indexar`. |
| `ModuleNotFoundError` | Activate the environment: `source .venv/bin/activate`. |

## Results

### Detector validation (tile level)

Pilot run `runs_nju/yolo11n_piloto` (YOLO11n, 2 epochs, `imgsz=640`, `batch=2`), validated on the tiles of the validation sequence:

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **1 (`best.pt`)** | **0.642** | **0.593** | **0.621** | **0.181** |
| 2 (`last.pt`) | 0.619 | 0.606 | 0.561 | 0.175 |

`best.pt` is the epoch with the highest Ultralytics fitness (0.1·mAP50 + 0.9·mAP50-95). With only 2 epochs (still inside the 3-epoch warm-up), the model is far from converged: localization is weak (mAP50-95 = 0.18) and the second epoch did not improve on the first.

### Detector evaluation on a full frame

`metricas.json` evaluates predictions on **1 full frame** after merging the tiles and removing duplicates with NMS (IoU 0.6), which is how the model would be used on gigapixel images:

| IoU threshold | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 0.50 | 8 | 41 | 18 | 0.163 | 0.308 | 0.213 |
| 0.75 | 1 | 48 | 25 | 0.020 | 0.038 | 0.027 |

- **Pedestrian** (IoU 0.5): precision 0.20, recall 0.31, F1 0.24.
- **Car**: 9 predictions and no cars in the ground truth, so all 9 are false positives (precision 0; recall undefined).
- **Bus / truck**: no ground-truth objects and no predictions in this frame.
- Deduplication removed most of the overlapping boxes (pedestrians 680 → 40, cars 342 → 9), but frame-level precision is still much lower than the tile-level figure. Objects cut by tile borders and the very short training are the likely causes. Since this is a single frame, these numbers are indicative only.

### Analysis agent on the validation set

`python agente.py exportar --fuente nju_yolo/images/val` produced **123 frames** (688 tiles, all from sequence `UAV-BASE-13_6`) with **1228 detections, all pedestrians** (mean confidence 0.34); no vehicles were detected. Indexing produced **599 documents**.

The agent (Gemini `gemini-3.5-flash`, `LLM_EFFORT=low`) was checked against answers computed directly from the JSON files:

| Check | Result |
|---|---|
| Total pedestrians (1228) | ✅ correct, cited `[resumen_global]` |
| Busiest grid cell of frame `…00000088` (`r3c3`, bottom-right, 5 pedestrians) | ✅ correct |
| Warns that tile-based counts have partial coverage | ✅ |
| Number of cars (0, no vehicles detected) | ✅ |
| Out-of-scope question (bicycles) answered without inventing data | ✅ |
| Busiest frame / heatmap chart | ⚠️ not completed: the provider returned `503 UNAVAILABLE` and `429 RESOURCE_EXHAUSTED` (free-tier quota); retry pending |

All completed answers used the retrieval tool before answering and cited their sources (4–38 s per answer). The test suite (`pytest tests`, 24 tests) passes.

### Next steps

- Train for more epochs (well beyond the 3-epoch warm-up) and with a larger batch; then re-run `exportar` and `indexar` so the agent reflects the new model.
- Evaluate on more full frames and more validation sequences (the current validation set has a single sequence and only pedestrians in the predictions).
- Improve cross-tile merging (e.g. intersection-over-smaller instead of IoU) to further reduce duplicate boxes cut by tile borders.

## Notes

- `Image.MAX_IMAGE_PIXELS` is disabled because the frames exceed Pillow's size limit.
- Hugging Face timeouts (`HF_HUB_DOWNLOAD_TIMEOUT`, `HF_HUB_ETAG_TIMEOUT`) are set so a half-open socket doesn't leave the download hanging.
- `.env` (API keys), `.venv/`, the dataset (`nju_yolo/`), model runs (`runs_nju/`, `*.pt`), the vector database (`data/chroma/`) and agent outputs (`resultados/`) are git-ignored.
- The agent sends only text (questions and retrieved summaries) to the LLM provider; images never leave your machine.
- The dataset and the model have their own licenses; check them before redistributing results.

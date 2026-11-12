# Object Detection in Gigapixel Images with YOLO (NJU-HARD)

Pilot project to detect **pedestrians, cars, buses and trucks** in the very high-resolution frames (~100–120 MP) of the [`RL-MIND/NJU-HARD-Detection`](https://huggingface.co/datasets/RL-MIND/NJU-HARD-Detection) dataset using YOLO11n (`ultralytics`).

Since the frames are too large to feed whole into the model, the pipeline **splits them into overlapping tiles** (1024 px, 128 px overlap), builds a dataset in YOLO format, trains the model, validates it, and lets you visualize its predictions.

## Repository structure

| File | What it does |
|---|---|
| `yolo.py` | Full pipeline: sequence scanning, parallel tiling, training, validation and a sample prediction. |
| `prediccion.py` | Draws the boxes predicted by the trained model, with confidence in %, on a validation image. |

## Requirements

- Linux with an **NVIDIA GPU and CUDA** (the script raises an error if no GPU is detected).
- Python 3.10+ and these libraries:

```bash
pip install torch ultralytics datasets pillow pyyaml tqdm
```

- Hugging Face access to stream the dataset.
- Enough RAM: each decoded frame takes ~350 MB in RGB and several are processed in parallel. If you run out of memory, lower `NUM_WORKERS` in `yolo.py`.

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

It also works in Google Colab: if detected, it mounts Google Drive (you can change `BASE_DIR` to `Path("/content")`).

### 2. Visualize predictions

Once the model is trained, without retraining:

```bash
python prediccion.py                  # picks a validation image with many detections
python prediccion.py path/image.jpg   # or uses the image you specify
```

It uses the most recent `best.pt` in `runs_nju/` and saves the result to `prediccion_ejemplo.jpg`, with each box labeled as `class NN%`. If none of the tested images has detections with confidence ≥ 0.25, it suggests lowering `CONF` (for example to 0.10).

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

Classes: `0: pedestrian`, `1: car`, `2: bus`, `3: truck`.

> Training is configured as a pilot (2 epochs, batch size 2). For a real training run, increase `epochs` and `batch` in `train_model()`.

## Results

🚧 **To be updated.** The YOLO model metrics (precision, recall, mAP50 and mAP50-95) are printed at the end of the validation step in `yolo.py`; they will be added here once the final results are available.

## Notes

- `Image.MAX_IMAGE_PIXELS` is disabled because the frames exceed Pillow's size limit.
- Hugging Face timeouts (`HF_HUB_DOWNLOAD_TIMEOUT`, `HF_HUB_ETAG_TIMEOUT`) are set so a half-open socket doesn't leave the download hanging.
- The dataset and the model have their own licenses; check them before redistributing results.

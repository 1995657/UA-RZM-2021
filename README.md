# UA-RZM: Same-epoch GNSS ZTD Spatial Modeling (ONNX Inference)

UA-RZM is a deep-learning model for **same-epoch spatial modeling** of Zenith Tropospheric Delay (ZTD).  
Given GNSS station ZTD observations at epoch *t* and user-specified target locations (stations or grid points) at the **same epoch**, UA-RZM estimates ZTD at those targets. It is designed for real-time use and **does not perform temporal forecasting**.

This repository provides:
- Pre-trained UA-RZM models in **ONNX** format (multi-altitude layers relative to DEM).
- Inference scripts to generate ZTD estimates at station locations and/or gridded ZTD products.
- Environment setup, execution instructions, input/output specifications.
- Basic performance statistics consistent with the accompanying paper.

> **Note on data privacy**  
> This repository can be used without distributing confidential raw station data. Users can run inference using their own station ZTD inputs.

---

## 1. Method summary

UA-RZM couples a U-Net–style encoder–decoder backbone with multi-head self-attention to aggregate non-local context.  
Key design points:
- **Physically meaningful coordinates:** latitude–longitude–elevation are explicit inputs to help learn terrain-dependent gradients.
- **Multi-scale representation:** the encoder increases feature resolution and the decoder compresses features to combine local gradients with regional structures.
- **Attention as adaptive weighting:** self-attention can be interpreted as a learned interpolation kernel that uses feature similarity (not distance alone).
- **Robustness to sparse/partial observations:** training uses random station masking (as described in the paper) to improve stability under degraded station geometries.

---

## 2. Quick start

### 2.1 Requirements
- Python >= 3.8 (recommended)
- OS: Windows/Linux (tested on Windows; Linux should work with the same dependencies)
- Key packages:
  - `onnxruntime`
  - `numpy`, `pandas`
  - `scikit-learn`
  - `torch` (optional; ONNX inference uses `onnxruntime`, but some helper scripts may import `torch`)

### 2.2 Install
```bash
pip install onnxruntime numpy pandas scikit-learn torch
```

### 2.3 Run inference
Place the required input files as described below, then run:
```bash
python FOR.py
```

Outputs will be written to the `Result/` directory, organized by height layer.

---

## 3. Repository layout (recommended)

```
UA-RZM_Model/
├── FOR.py
├── data_loader.py
├── data_get.py
├── UA-RZM/
│   ├── ua-rzm-400.onnx
│   ├── ua-rzm-300.onnx
│   ├── ...
│   └── ua-rzm10000.onnx
├── Grid/
│   └── dem{H}/dem{H}.txt
├── Verification_Data/
│   ├── YYYY_DDD_map_trop00.txt
│   └── ...
└── Result/
    └── dem{H}/
```

- **ONNX models:** `UA-RZM/ua-rzm{H}.onnx` correspond to altitude layers (meters relative to DEM).  
- **Grid definition:** `Grid/dem{H}/dem{H}.txt` defines target points (1°×1° in the released configuration).  
- **Station inputs:** `Verification_Data/` contains same-epoch station files.

---

## 4. Inputs and outputs

### 4.1 Station input files (`Verification_Data/`)
**Naming convention:** `YYYY_DDD_map_tropHH.txt` (e.g., `2022_001_map_trop00.txt`)  
**Columns (space-separated):**
1. `Longitude` (deg)
2. `Latitude` (deg)
3. `GeodeticHeight` (m)
4. `ZTD` (m)

**Important:**
- Keep the station order unchanged within each file.
- Coordinates should be geodetic (lat/lon/height). Heights should be consistent with the model configuration.

### 4.2 Grid definition files (`Grid/dem{H}/dem{H}.txt`)
**Columns (space-separated):**
1. `Longitude` (deg)
2. `Latitude` (deg)
3. `GeodeticHeight` (m)
4. `ZTD` (m)

**Note:** the ZTD column in grid definition files is not used for inference and can be set to 0.

### 4.3 Outputs (`Result/dem{H}/`)
For each input station file, UA-RZM writes a corresponding output file under `Result/dem{H}/`.  
**Output columns (space-separated):**
1. `Longitude` (deg)
2. `Latitude` (deg)
3. `GeodeticHeight` (m)
4. `ZTD_est` (m)

---

## 5. Height layers

The default height layers (meters relative to DEM) are:
```
-400, -300, -200, -150, -50, 0, 50, 100, 200, 300, 500, 1000, 2000, 5000, 7500, 10000
```
Each layer uses a dedicated ONNX model file `ua-rzm{H}.onnx` and a corresponding grid definition file `Grid/dem{H}/dem{H}.txt`.

---

## 6. Basic performance (as reported in the paper)

These values are provided as a quick reference and should be interpreted together with the full experimental setup in the paper.

- **Station-wise (China, independent year test):** RMSE 1.37 cm, MAE 0.92 cm.  
- **Surface gridded product (China):** RMSE 1.25 cm, MAE 0.90 cm, R² 0.9990 (compared with VMF3_FC as reported); bias is slightly larger than VMF3_FC.

---

## 7. Reproducibility notes

- UA-RZM performs **same-epoch spatial mapping**. It is not a forecasting model.
- Inference requires only same-epoch station ZTD and coordinates plus target coordinates. No ERA5 fields are needed at inference time.
- ERA5 is used in training as supervision and in evaluation as reference, as described in the paper.

---

## 8. Data

The products related to the model section of this paper are available via the following link:Link: https://pan.baidu.com/s/1VfItajc5-nJN8L4fsy0jQA?pwd=uarzExtraction code: uarz


## 9. Citation

If you use this code or the released models, please cite the accompanying paper:



---

## 10. License and disclaimer

For research use only. The software and models are provided “as is” without any warranty.  
Please ensure compliance with your data-sharing policies when running the model on proprietary station observations.

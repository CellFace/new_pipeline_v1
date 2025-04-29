# DHM Image Processing GUI Application

_Real-time Image Capture ▶ Reconstruction ▶ Object Detection ▶ Result Storage_

---

## 0 ▪ Workflow

1. **Image Capture:** Images are captured using the **Spinnaker SDK**, which provides a robust interface for camera operations and image acquisition.
2. **Image Reconstruction:** Captured images are enhanced using an **AI-based reconstructor** for improved quality and further processing.
3. **Object Detection:** A YOLO-based object detection model identifies and classifies objects within the reconstructed images.
4. **Data Storage:** Final results are stored in a **containerized HDF5 file**, ensuring organized access to all outputs.

---

## 1 ▪ Installation

> **Note:** Use **Python 3.8** and create a new virtual environment with it.

### Step-by-step Instructions

```bash
# 1️⃣ Create and activate environment
python -m venv .venv
.venv\Scripts\activate   # for Windows
source .venv/bin/activate  # for Linux/macOS

# 2️⃣ Install Spinnaker SDK (PySpin)
python -m pip install spinnaker_python-2.7.0.128-cp38-cp38-win_amd64.whl

# 3️⃣ Install general dependencies
pip install -r requirements.txt

# 4️⃣ Install YOLO
pip install ultralytics

# 5️⃣ Install PyTorch
# Visit https://pytorch.org/ and select your configuration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch
python -c "import torch; print(torch.__version__)"
```

---

## 2 ▪ Usage

### A. Starting the Application

- Navigate to `dhm/views/`
- Run `main_view.py`
- If installation is correct, the **GUI** should appear

### B. Configuration Setup

1. Click the **“Configuration”** button.
2. In the popup form:
   - Enter values for required parameters (e.g., `pixel_format`, `exposure_seconds`)
   - Provide paths for:
     - YOLO weight file
     - AI reconstructor models
     - Output folder
3. Click **“Save Configuration”** to store settings.

### C. Analyzing h5 Capture Files

1. Click **“Process Capture File”**
2. Select `.h5` file using **“Select File”**
3. Choose reconstruction method: **Ovizio** or **AI**
4. Click **“Process File”**
5. If successful, results will be saved as a `.h5` container file

### D. Analyzing Reconstructed PNG Images

1. Provide the folder path to PNG phase images
2. Click **“Process File”**
3. Results will be saved as `.h5` container file (if successful)

### E. Previewing Microscope Images

- Click **“Preview Images”** to start real-time hologram image preview

### F. Configure Paths

Ensure the following are correctly updated in your config:

- `VALID_H5_PATH` → path to H5 capture file
- `INPUT_PATH` → path to Ovizio PNGs
- `VALID_IMG_PATH`, `VALID_WEIGHT_PATH` → for detection input

---

## 3 ▪ Notes

- Double-check **device compatibility** when installing PyTorch (e.g., CUDA version)
- Ensure **Spinnaker SDK** is properly installed and the `.whl` matches Python version

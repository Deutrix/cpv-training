# RunPod Deployment Guide

This guide explains how to train the CPV Detector on RunPod using a GPU instance.

## 1. Create a Pod

1.  Go to [RunPod.io](https://www.runpod.io/) and log in.
2.  Click **Deploy** on a GPU instance (e.g., RTX 3090 or RTX 4090 are good choices).
3.  Select a template: **RunPod PyTorch** (usually default) is recommended as it comes with CUDA and Python pre-installed.
4.  Ensure you have enough disk space (at least 20GB-30GB for dataset and models).
5.  Click **Deploy**.

## 2. Connect to the Pod

1.  Once the pod is running, click **Connect**.
2.  Select **Start Web Terminal** (or use Jupyter Lab if you prefer).

## 3. Upload Files

You can upload files via Jupyter Lab (easiest for small number of files) or use `scp` / `rsync`.

**Files to upload:**
-   `train.py`
-   `predict.py`
-   `training_data.csv`
-   `cpv_codes.csv`
-   `requirements.txt`

**Using Jupyter Lab:**
1.  Click **Connect** -> **Connect to Jupyter Lab**.
2.  In the file browser (left sidebar), click the **Upload Files** button (up arrow).
3.  Select the 5 files listing above from your computer.

## 4. Install Dependencies

In the Web Terminal (or Jupyter Terminal):

```bash
pip install -r requirements.txt
```

## 5. Run Training

To start the training process:

```bash
python train.py
```

*Note: If you want to run in the background (so it doesn't stop if you close the browser), you can use `nohup`:*

```bash
nohup python train.py > training.log 2>&1 &
```
And check progress with `tail -f training.log`.

## 6. Run Inference / Prediction

Once training is complete (model saved in `models/cpv-decoder`), you need to build the embeddings index first:

```bash
python build_embeddings.py
```

Then you can test it:

```bash
python predict.py "nabavka računara"
```

## 7. Run API (FastAPI)

To run the API server on RunPod (ensure you expose port 8000 in Pod settings if accessing externally):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 8. Download Model

To save your trained model back to your local machine:
1.  Go to Jupyter Lab.
2.  Navigate to `models/cpv-decoder`.
3.  Right-click the folder (or zip it first) and select **Download**.
    *To zip via terminal:*
    ```bash
    zip -r cpv_model.zip models/cpv-decoder
    ```

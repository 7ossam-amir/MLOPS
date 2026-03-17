# MLOPS_A2 Kaggle GPU Run (Single Entrypoint)

This project uses one runner file: `main.py`.

No `run_kaggle.py` and no builder script are needed.

## Required files

Keep these in your project:

- `main.py`
- `config.yaml`
- `src/`
- `digit-recognizer/train.csv` (or provide a Kaggle dataset that contains `train.csv`)

`main.py` reads epochs from `config.yaml` (`training.epochs`).

## One-time setup

1. Install Kaggle CLI:

```powershell
pip install kaggle
```

2. Get your Kaggle API key from Kaggle account settings.

3. Set auth in PowerShell (use `KAGGLE_KEY`, not `KAGGLE_API_TOKEN`):

```powershell
$env:KAGGLE_USERNAME="hossamamir"
$env:KAGGLE_KEY="<your_kaggle_key>"
```

4. Confirm kernel metadata in `kernel-metadata.json`:

```json
{
  "id": "hossamamir/mlops-a2-gpu",
  "code_file": "main.py",
  "enable_gpu": true
}
```

## Run every time (PowerShell)

```powershell
cd C:\Users\Hossam\Desktop\MLOPS_A2

$env:KAGGLE_USERNAME="hossamamir"
$env:KAGGLE_KEY="<your_kaggle_key>"

python .\scripts\sync_run_kaggle_embed.py
kaggle kernels push -p .
kaggle kernels status hossamamir/mlops-a2-gpu
```

## See epochs live

```powershell
$kernel="hossamamir/mlops-a2-gpu"
while ($true) {
  kaggle kernels output $kernel -p .\kaggle_output --force | Out-Null

  if (Test-Path .\kaggle_output\mlops-a2-gpu.log) {
    Get-Content .\kaggle_output\mlops-a2-gpu.log |
      Select-String "Using device|Epoch|Traceback|ERROR" |
      Select-Object -Last 20
  }

  $s = kaggle kernels status $kernel
  $s

  if ($s -match "COMPLETE|ERROR") { break }
  Start-Sleep 20
}
```

## Download outputs

```powershell
kaggle kernels output hossamamir/mlops-a2-gpu -p .\kaggle_output --force
```

Expected outputs:

- `kaggle_output/outputs/generated_digits.png`
- `kaggle_output/outputs/loss_curves.png`
- `kaggle_output/outputs/real_vs_fake.png`

## MLflow (Local Web UI)

MLflow logging is controlled from `config.yaml`:

```yaml
mlflow:
  enabled: true
  tracking_uri: mlruns
  experiment_name: mlops_a2_gan
  run_name: ""
```

Run training locally:

```powershell
cd C:\Users\Hossam\Desktop\MLOPS_A2
pip install -r requirements.txt
python main.py --config config.yaml
```

Start MLflow UI:

```powershell
mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000
```

Open:

- `http://127.0.0.1:5000`

You will see:

- One run per training execution
- Epoch metrics (`generator_loss`, `discriminator_loss`)
- Final metrics (`real_score`, `fake_score`)
- Logged plot artifacts (`generated_digits.png`, `loss_curves.png`, `real_vs_fake.png`)
- Logged epoch text file (`logs/training_log.txt`)

Note: The **Traces** tab is for MLflow tracing instrumentation (`@mlflow.trace`). This project logs training runs/metrics/artifacts, so use the **Runs** view.

If you run through Kaggle (`run_kaggle.py`), MLflow logs are saved under `outputs/mlruns` in the kernel output files.

To keep all Kaggle experiments (without overwrite), archive each run to a timestamped folder:

```powershell
.\scripts\pull_and_archive_mlflow.ps1
```

The script auto-detects older Kaggle CLI versions that do not support `--file-pattern` and falls back to full output download.

Then open the combined archive:

```powershell
mlflow ui --backend-store-uri .\mlruns_archive --host 127.0.0.1 --port 5001
```

## How to confirm GPU was used

Check the log line:

- `Using device: cuda` -> GPU used
- `Using device: cpu` -> CPU used

## Why `mlops-a2-gpu` is the name

It comes from `kernel-metadata.json`:

```json
"id": "hossamamir/mlops-a2-gpu"
```

## Common errors

`You must authenticate before you can call the Kaggle API.`

- Fix: set `KAGGLE_USERNAME` and `KAGGLE_KEY` correctly.

`Could not find main.py/config.yaml/src in Kaggle runtime.`

- Cause: script kernels may run without your full local project tree.
- Fix: attach your project files as a Kaggle dataset source, or switch to a repo-clone workflow.

`404 ... /train.csv` (ngrok)

- Cause: URL/path is wrong or local file server is not serving that path.
- Note: this single-file Kaggle flow does not require ngrok.

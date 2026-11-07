# Weights

Model weights are **not stored in this git repository** because of size and verification status.

- Download the demo weight from the project site, or train your own with `start_training_cli.bat` / `start_cli.sh`.
- Place weights under `models_weights/<arch>/`, for example `models_weights/UNet/unet_best.pth`.
- Training outputs are `verified=false` drafts unless independently evaluated. The WebUI marks unverified weights explicitly.
- The 2026-11 patient-level results in the README come from the 9-hour study; they are not a claim that the demo weight is clinically validated.

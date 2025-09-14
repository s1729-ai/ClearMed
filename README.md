Medical Q&A: Training on medquad.csv

Quick Start (Google Colab recommended)

1. Upload this folder to Colab or mount via Google Drive.
2. Install deps:
```
pip install -r requirements.txt
```
3. Prepare data:
```
python data_prep.py
```
4. Train LoRA (needs GPU, ~16GB VRAM with 4-bit offload):
```
python train_lora.py
```
5. Inference:
```
python infer.py
```

Notes

- Base model: `mistralai/Mistral-7B-Instruct-v0.2` (change in `train_lora.py` if desired).
- Outputs saved to `outputs/lora-medqa/adapter`.
- Data split: 95% train / 5% val from `medquad.csv`.

Safety

This project is for research/education. It is not a substitute for professional medical advice.



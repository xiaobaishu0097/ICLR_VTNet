# VTNet: Visual Transformer Network for Object Goal Navigation


## Install
```
pip install -r requirements.txt
```

## Training

```python
python main.py --gpu-ids 0 --workers 4 --model VTNetModel --detr --title a3c_vtnet --work-dir ./work_dirs/
```

## Pre-training
```
python main_pretraining.py --gpu-ids 0 --workers 4 --model PreTrainedVisualTransformer --detr --title a3c --work-dir ./work_dirs/
```

## Testing

```python
python full_eval.py --gpu-ids 0 --detr --save-model-dir {SAVE_MODEL_DIR} --results-json ./result.json --model VTNetModel --title a3c_previstrans_base
```
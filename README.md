### Train
```bash
python main.py --task=vrp10
```
### Inference

```bash
python main.py --task=vrp10 --is_train=False --model_dir=./path_to_your_saved_checkpoint
```

### Logs
All logs are stored in ``result.txt`` file stored in ``./logs/task_date_time`` directory.
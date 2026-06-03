# Data Availability

This repository provides source code, representative occlusion masks,
visualization scripts, and manuscript figures for the study.

The complete traffic-sign image-text dataset and trained model checkpoints are
not stored in this GitHub repository because they are large research artifacts.
They can be made available by the corresponding author upon reasonable request,
subject to applicable data-use and storage constraints.

To reproduce the evaluation pipeline, place local files in this structure:

```text
data/
  val_images/
  occluded_images/
  val_metadata.json
checkpoints/
  model_best.pth.tar
```

The expected metadata format is:

```json
[
  {
    "image_path": "image_001.jpg",
    "captions": ["traffic sign caption"],
    "category": "speed_limit"
  }
]
```

After preparing the local data and checkpoint files, run:

```bash
python scripts/evaluate_oata_tent.py \
  --model_path checkpoints/model_best.pth.tar \
  --dataset_path data/occluded_images \
  --metadata_path data/val_metadata.json \
  --single \
  --use_occ \
  --use_ent
```


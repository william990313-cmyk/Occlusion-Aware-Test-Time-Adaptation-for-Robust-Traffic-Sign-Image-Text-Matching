# Data Directory

Place evaluation images and metadata here before running the scripts.

Expected layout:

```text
data/
  val_images/
    image_001.jpg
    image_002.jpg
  occluded_images/
    image_001.jpg
    image_002.jpg
  val_metadata.json
```

Each metadata entry should contain at least:

```json
{
  "image_path": "image_001.jpg",
  "captions": ["traffic sign caption"],
  "category": "speed_limit"
}
```

The complete traffic-sign image-text dataset is not included in this repository.
See `DATA_AVAILABILITY.md` for access details.

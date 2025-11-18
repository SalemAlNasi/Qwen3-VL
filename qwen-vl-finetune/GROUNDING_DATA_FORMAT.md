# Grounding Task Data Format

## JSON Format for Training

### Bounding Box (bbox_2d)

```json
{
    "image": "path/to/image.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nLocate the house in this image."
        },
        {
            "from": "gpt",
            "value": "{\"bbox_2d\": [135, 114, 1016, 672], \"label\": \"house\"}"
        }
    ]
}
```

### Pointing (point_2d)

```json
{
    "image": "path/to/image.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nLocate all people in this image."
        },
        {
            "from": "gpt",
            "value": "[{\"point_2d\": [346, 328], \"label\": \"person\"}, {\"point_2d\": [238, 158], \"label\": \"person\"}]"
        }
    ]
}
```

### Pointing with Attributes

```json
{
    "image": "path/to/image.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nLocate all people with their roles and shirt colors."
        },
        {
            "from": "gpt",
            "value": "[{\"point_2d\": [346, 328], \"label\": \"person\", \"role\": \"referee\", \"shirt_color\": \"black\"}, {\"point_2d\": [238, 158], \"label\": \"person\", \"role\": \"player\", \"shirt_color\": \"red\"}]"
        }
    ]
}
```

## Coordinate Systems

### Qwen3-VL (Recommended)
- **Format**: Relative coordinates (0-1000 range)
- **Example**: `[125, 250]` or `[125, 250, 500, 750]`
- **Advantage**: Scale-invariant, no adjustment needed

### Qwen2.5-VL
- **Format**: Absolute pixel coordinates for original image
- **Example**: `[100, 200]` or `[100, 200, 400, 600]`
- **Note**: Automatically adjusted during training when images are resized

### Qwen2-VL
- **Format**: Relative coordinates (0-1000 range)
- **Same as Qwen3-VL**

## Coordinate Format

- **bbox_2d**: `[x1, y1, x2, y2]` - Top-left and bottom-right corners
- **point_2d**: `[x, y]` - Single point location

## Notes

- Coordinates are automatically adjusted during training based on `min_pixels` and `max_pixels` settings
- Use `<image>` token in human message to indicate image position
- JSON format for `gpt` responses (use double quotes)
- Can mix bbox and point tasks in the same dataset

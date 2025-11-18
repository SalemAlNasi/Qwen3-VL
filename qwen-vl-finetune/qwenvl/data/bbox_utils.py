"""
Utility functions for adjusting bounding box coordinates during online training.
Handles coordinate transformation when images are dynamically resized.
"""

import re
import json
from typing import Dict, Tuple, Any
from PIL import Image


def smart_resize_for_bbox(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> Tuple[int, int]:
    """
    Calculate resized dimensions (same logic as vision_process.smart_resize).
    
    Args:
        height: Original image height
        width: Original image width  
        factor: Resize factor (typically 28 for patch_size=14 * merge_size=2)
        min_pixels: Minimum total pixels allowed
        max_pixels: Maximum total pixels allowed
        
    Returns:
        Tuple of (resized_height, resized_width)
    """
    import math
    
    MAX_RATIO = 200
    
    def round_by_factor(number: int, factor: int) -> int:
        return round(number / factor) * factor
    
    def ceil_by_factor(number: int, factor: int) -> int:
        return math.ceil(number / factor) * factor
    
    def floor_by_factor(number: int, factor: int) -> int:
        return math.floor(number / factor) * factor
    
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    
    return h_bar, w_bar


def adjust_bbox_coordinates(
    bbox: list,
    orig_height: int,
    orig_width: int,
    resized_height: int,
    resized_width: int,
    is_relative: bool = True
) -> list:
    """
    Adjust bounding box coordinates based on image resize.
    
    Args:
        bbox: Original bbox coordinates [x1, y1, x2, y2]
        orig_height: Original image height
        orig_width: Original image width
        resized_height: Resized image height
        resized_width: Resized image width
        is_relative: True for Qwen2/Qwen3 (0-1000), False for Qwen2.5 (pixels)
        
    Returns:
        Adjusted bbox coordinates
    """
    x1, y1, x2, y2 = bbox
    
    if is_relative:
        # Qwen2-VL and Qwen3-VL use relative coordinates (0-1000 range)
        # These should already be in relative format, but we verify/convert
        if max(bbox) > 1000:
            # If coordinates are absolute, convert to relative
            print(f"[WARNING] BBox {bbox} appears to be absolute (>1000), "
                  f"converting to relative based on original dimensions")
            new_x1 = int((x1 / orig_width) * 1000)
            new_y1 = int((y1 / orig_height) * 1000)
            new_x2 = int((x2 / orig_width) * 1000)
            new_y2 = int((y2 / orig_height) * 1000)
        else:
            # Already in relative format, no adjustment needed
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
    else:
        # Qwen2.5-VL uses absolute coordinates for resized image
        # Scale from original absolute to resized absolute
        scale_x = resized_width / orig_width
        scale_y = resized_height / orig_height
        
        new_x1 = int(x1 * scale_x)
        new_y1 = int(y1 * scale_y)
        new_x2 = int(x2 * scale_x)
        new_y2 = int(y2 * scale_y)
    
    return [new_x1, new_y1, new_x2, new_y2]


def parse_and_adjust_bbox_in_text(
    text: str,
    orig_height: int,
    orig_width: int,
    resized_height: int,
    resized_width: int,
    is_relative: bool = True,
    debug: bool = False
) -> str:
    """
    Parse text containing bbox_2d coordinates and adjust them.
    
    Args:
        text: Text containing JSON with bbox_2d field
        orig_height: Original image height
        orig_width: Original image width
        resized_height: Resized image height  
        resized_width: Resized image width
        is_relative: True for Qwen2/Qwen3 (0-1000), False for Qwen2.5 (pixels)
        debug: If True, print adjustment details
        
    Returns:
        Text with adjusted bbox coordinates
    """
    # Pattern to match bbox_2d in JSON format
    # Matches: "bbox_2d": [x1, y1, x2, y2]
    bbox_pattern = r'"bbox_2d"\s*:\s*\[([^\]]+)\]'
    
    def replace_bbox(match):
        coords_str = match.group(1)
        try:
            # Parse coordinates
            coords = [int(x.strip()) for x in coords_str.split(',')]
            if len(coords) != 4:
                return match.group(0)  # Return unchanged if not 4 coordinates
            
            # Adjust coordinates
            adjusted_coords = adjust_bbox_coordinates(
                coords,
                orig_height,
                orig_width,
                resized_height,
                resized_width,
                is_relative
            )
            
            if debug:
                print(f"[BBOX ADJUSTMENT] Original: {coords}, "
                      f"Adjusted: {adjusted_coords}")
            
            # Return adjusted string
            return f'"bbox_2d": [{", ".join(map(str, adjusted_coords))}]'
        except (ValueError, IndexError) as e:
            # If parsing fails, return original
            if debug:
                print(f"[BBOX ADJUSTMENT ERROR] {e}")
            return match.group(0)
    
    # Replace all bbox_2d occurrences
    adjusted_text = re.sub(bbox_pattern, replace_bbox, text)
    return adjusted_text


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get image dimensions without fully loading it into memory.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (width, height)
    """
    with Image.open(image_path) as img:
        return img.size  # Returns (width, height)


def should_adjust_coordinates(conversation: Dict[str, Any]) -> bool:
    """
    Check if a conversation contains bbox coordinates that need adjustment.
    
    Args:
        conversation: Conversation dict with 'value' field
        
    Returns:
        True if bbox_2d is found in the conversation
    """
    value = conversation.get("value", "")
    return "bbox_2d" in value

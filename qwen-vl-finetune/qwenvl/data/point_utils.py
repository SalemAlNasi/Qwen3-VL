"""
Pointing coordinate utilities for Qwen VL models.

Provides functions for:
- Converting pixel coordinates to Qwen format
- Adjusting point coordinates when images are resized (for Qwen2.5-VL)
- Parsing and modifying point_2d in text

For Qwen3-VL (relative 0-1000 coordinates): No adjustment needed
For Qwen2.5-VL (absolute pixel coordinates): Adjustment required on resize

Author: Auto-generated
Date: 2025
"""

import re
import json
from typing import Tuple, List, Dict, Any
from PIL import Image


def pixel_to_qwen3vl_point(
    pixel_x: int,
    pixel_y: int,
    image_width: int,
    image_height: int
) -> List[int]:
    """
    Convert pixel coordinates to Qwen3-VL relative format (0-1000 range).
    
    Args:
        pixel_x: X coordinate in pixels
        pixel_y: Y coordinate in pixels
        image_width: Original image width in pixels
        image_height: Original image height in pixels
        
    Returns:
        Point coordinates in [x, y] format with 0-1000 range
        
    Example:
        >>> pixel_to_qwen3vl_point(100, 150, 800, 600)
        [125, 250]
    """
    normalized_x = int((pixel_x / image_width) * 1000)
    normalized_y = int((pixel_y / image_height) * 1000)
    return [normalized_x, normalized_y]


def qwen3vl_to_pixel_point(
    normalized_x: int,
    normalized_y: int,
    image_width: int,
    image_height: int
) -> Tuple[int, int]:
    """
    Convert Qwen3-VL format back to pixel coordinates.
    
    Args:
        normalized_x: X coordinate in 0-1000 range
        normalized_y: Y coordinate in 0-1000 range
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Pixel coordinates as (x, y) tuple
        
    Example:
        >>> qwen3vl_to_pixel_point(125, 250, 800, 600)
        (100, 150)
    """
    pixel_x = int((normalized_x / 1000) * image_width)
    pixel_y = int((normalized_y / 1000) * image_height)
    return (pixel_x, pixel_y)


def adjust_point_coordinates(
    point: list,
    original_height: int,
    original_width: int,
    resized_height: int,
    resized_width: int,
    is_relative: bool = True
) -> list:
    """
    Adjust point coordinates after image resize.
    
    Args:
        point: Original point [x, y]
        original_height: Original image height in pixels
        original_width: Original image width in pixels
        resized_height: Resized image height in pixels
        resized_width: Resized image width in pixels
        is_relative: True for Qwen3-VL (0-1000 range), False for Qwen2.5-VL (pixels)
        
    Returns:
        Adjusted point coordinates [x, y]
        
    Note:
        For Qwen3-VL (relative coordinates), this returns the original point unchanged.
        For Qwen2.5-VL (absolute coordinates), adjusts based on resize ratios.
    """
    x, y = point
    
    if is_relative:
        # Qwen3-VL: relative coordinates are scale-invariant
        # Check if coords appear to be relative (0-1000 range)
        if max(point) <= 1000:
            return [x, y]
        else:
            # Appears to be absolute, convert to relative
            print(f"[WARNING] Point {point} appears to be absolute (>1000), "
                  f"converting to relative based on original dimensions")
            return [
                int((x / original_width) * 1000),
                int((y / original_height) * 1000)
            ]
    else:
        # Qwen2.5-VL: absolute pixel coordinates need adjustment
        ratio_w = resized_width / original_width
        ratio_h = resized_height / original_height
        
        adjusted_x = int(x * ratio_w)
        adjusted_y = int(y * ratio_h)
        
        return [adjusted_x, adjusted_y]


def parse_and_adjust_point_in_text(
    text: str,
    original_height: int,
    original_width: int,
    resized_height: int,
    resized_width: int,
    is_relative: bool = True,
    debug: bool = False
) -> str:
    """
    Parse text containing point_2d coordinates and adjust them.
    
    Args:
        text: Text containing JSON with point_2d field(s)
        original_height: Original image height
        original_width: Original image width
        resized_height: Resized image height
        resized_width: Resized image width
        is_relative: True for Qwen3-VL, False for Qwen2.5-VL
        debug: If True, print adjustment details
        
    Returns:
        Text with adjusted point coordinates
        
    Example:
        >>> text = '{"point_2d": [100, 150], "label": "person"}'
        >>> parse_and_adjust_point_in_text(text, 600, 800, 450, 600, is_relative=False)
        '{"point_2d": [75, 112], "label": "person"}'
    """
    # Pattern to match point_2d in JSON format
    # Matches: "point_2d": [x, y]
    point_pattern = r'"point_2d"\s*:\s*\[([^\]]+)\]'
    
    def replace_point(match):
        coords_str = match.group(1)
        try:
            # Parse coordinates
            coords = [int(float(c.strip())) for c in coords_str.split(',')]
            if len(coords) != 2:
                return match.group(0)  # Return original if not 2 coords
            
            # Adjust coordinates
            adjusted_coords = adjust_point_coordinates(
                coords,
                original_height,
                original_width,
                resized_height,
                resized_width,
                is_relative
            )
            
            if debug:
                print(f"[POINT ADJUSTMENT] Original: {coords}, "
                      f"Adjusted: {adjusted_coords}")
            
            # Return adjusted point in same format
            return f'"point_2d": [{adjusted_coords[0]}, {adjusted_coords[1]}]'
        except Exception as e:
            if debug:
                print(f"[POINT ADJUSTMENT ERROR] {e}")
            return match.group(0)  # Return original on error
    
    # Replace all point_2d occurrences
    adjusted_text = re.sub(point_pattern, replace_point, text)
    
    return adjusted_text


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get image dimensions without fully loading the image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (width, height)
        
    Note:
        This is a fast operation that only reads the image header.
    """
    with Image.open(image_path) as img:
        return img.size  # Returns (width, height)


def should_adjust_point_coordinates(conversation: dict) -> bool:
    """
    Check if a conversation contains point coordinates that need adjustment.
    
    Args:
        conversation: A conversation dict with 'from' and 'value' keys
        
    Returns:
        True if point_2d is found in the conversation
        
    Example:
        >>> conv = {'from': 'gpt', 'value': '{"point_2d": [100, 150]}'}
        >>> should_adjust_point_coordinates(conv)
        True
    """
    value = conversation.get('value', '')
    return "point_2d" in value


def batch_convert_pixels_to_qwen3vl(
    points: List[Dict[str, Any]],
    image_width: int,
    image_height: int
) -> List[Dict[str, Any]]:
    """
    Convert a batch of pixel-coordinate points to Qwen3-VL format.
    
    Args:
        points: List of point dicts with 'x', 'y' pixel coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        List of dicts with 'point_2d' in Qwen3-VL format plus other attributes
        
    Example:
        >>> points = [
        ...     {'x': 100, 'y': 150, 'label': 'person', 'role': 'player'},
        ...     {'x': 200, 'y': 250, 'label': 'person', 'role': 'referee'}
        ... ]
        >>> batch_convert_pixels_to_qwen3vl(points, 800, 600)
        [
            {'point_2d': [125, 250], 'label': 'person', 'role': 'player'},
            {'point_2d': [250, 417], 'label': 'person', 'role': 'referee'}
        ]
    """
    converted_points = []
    
    for point in points:
        # Convert coordinates
        qwen_coords = pixel_to_qwen3vl_point(
            point['x'], point['y'],
            image_width, image_height
        )
        
        # Create new dict with converted coords
        converted_point = {'point_2d': qwen_coords}
        
        # Copy all other attributes
        for key, value in point.items():
            if key not in ['x', 'y']:
                converted_point[key] = value
        
        converted_points.append(converted_point)
    
    return converted_points


def validate_point_format(point_dict: dict, is_relative: bool = True) -> Tuple[bool, str]:
    """
    Validate a point dictionary format.
    
    Args:
        point_dict: Dictionary containing point_2d and other fields
        is_relative: True for Qwen3-VL (0-1000), False for Qwen2.5-VL (any range)
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Example:
        >>> validate_point_format({'point_2d': [500, 500], 'label': 'person'})
        (True, '')
        >>> validate_point_format({'point_2d': [1500, 500], 'label': 'person'})
        (False, 'X coordinate 1500 out of range [0, 1000]')
    """
    # Check required fields
    if 'point_2d' not in point_dict:
        return False, "Missing 'point_2d' field"
    
    if 'label' not in point_dict:
        return False, "Missing 'label' field"
    
    # Check point_2d format
    point = point_dict['point_2d']
    if not isinstance(point, list) or len(point) != 2:
        return False, f"point_2d must be a list of 2 values, got {point}"
    
    # Check coordinate types
    try:
        x, y = int(point[0]), int(point[1])
    except (ValueError, TypeError):
        return False, f"Coordinates must be numbers, got {point}"
    
    # Check coordinate ranges for relative format
    if is_relative:
        if not (0 <= x <= 1000):
            return False, f"X coordinate {x} out of range [0, 1000]"
        if not (0 <= y <= 1000):
            return False, f"Y coordinate {y} out of range [0, 1000]"
    
    return True, ""


if __name__ == "__main__":
    # Example usage and tests
    print("=== Point Coordinate Utilities ===")
    print()
    
    # Test 1: Pixel to Qwen3-VL conversion
    print("Test 1: Pixel to Qwen3-VL conversion")
    pixel_point = (100, 150)
    img_size = (800, 600)
    qwen_point = pixel_to_qwen3vl_point(*pixel_point, *img_size)
    print(f"  Pixel coords: {pixel_point} in {img_size[0]}x{img_size[1]} image")
    print(f"  Qwen3-VL coords: {qwen_point}")
    print()
    
    # Test 2: Qwen3-VL to pixel conversion
    print("Test 2: Qwen3-VL to pixel conversion")
    qwen_coords = (125, 250)
    pixel_coords = qwen3vl_to_pixel_point(*qwen_coords, *img_size)
    print(f"  Qwen3-VL coords: {qwen_coords}")
    print(f"  Pixel coords: {pixel_coords} in {img_size[0]}x{img_size[1]} image")
    print()
    
    # Test 3: Relative coordinate adjustment (should be unchanged)
    print("Test 3: Qwen3-VL coordinate adjustment (relative)")
    original_point = [500, 500]
    adjusted = adjust_point_coordinates(
        original_point,
        original_height=600,
        original_width=800,
        resized_height=450,
        resized_width=600,
        is_relative=True
    )
    print(f"  Original (relative): {original_point}")
    print(f"  After resize: {adjusted} (should be unchanged)")
    print()
    
    # Test 4: Absolute coordinate adjustment
    print("Test 4: Qwen2.5-VL coordinate adjustment (absolute)")
    original_point = [400, 300]
    adjusted = adjust_point_coordinates(
        original_point,
        original_height=600,
        original_width=800,
        resized_height=450,
        resized_width=600,
        is_relative=False
    )
    print(f"  Original (absolute): {original_point} in 800x600 image")
    print(f"  After resize to 600x450: {adjusted}")
    print()
    
    # Test 5: Parse and adjust in text
    print("Test 5: Parse and adjust point in JSON text")
    text = '{"point_2d": [400, 300], "label": "person", "role": "player"}'
    adjusted_text = parse_and_adjust_point_in_text(
        text,
        original_height=600,
        original_width=800,
        resized_height=450,
        resized_width=600,
        is_relative=False,
        debug=True
    )
    print(f"  Original text: {text}")
    print(f"  Adjusted text: {adjusted_text}")
    print()
    
    # Test 6: Batch conversion
    print("Test 6: Batch pixel to Qwen3-VL conversion")
    pixel_points = [
        {'x': 100, 'y': 150, 'label': 'person', 'role': 'player', 'shirt_color': 'red'},
        {'x': 200, 'y': 250, 'label': 'person', 'role': 'referee', 'shirt_color': 'black'}
    ]
    qwen_points = batch_convert_pixels_to_qwen3vl(pixel_points, 800, 600)
    print(f"  Input (pixel coordinates):")
    for p in pixel_points:
        print(f"    {p}")
    print(f"  Output (Qwen3-VL format):")
    for p in qwen_points:
        print(f"    {p}")
    print()
    
    # Test 7: Validation
    print("Test 7: Point format validation")
    valid_point = {'point_2d': [500, 500], 'label': 'person'}
    invalid_point1 = {'point_2d': [1500, 500], 'label': 'person'}
    invalid_point2 = {'label': 'person'}
    
    is_valid, error = validate_point_format(valid_point)
    print(f"  Valid point: {is_valid}, Error: '{error}'")
    
    is_valid, error = validate_point_format(invalid_point1)
    print(f"  Invalid point (out of range): {is_valid}, Error: '{error}'")
    
    is_valid, error = validate_point_format(invalid_point2)
    print(f"  Invalid point (missing field): {is_valid}, Error: '{error}'")
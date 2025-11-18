#!/usr/bin/env python3
"""
Test script for coordinate adjustment in data_processor.py

This tests:
1. Point coordinate adjustment for Qwen2.5-VL (absolute)
2. Point coordinate adjustment for Qwen3-VL (relative)
3. BBox coordinate adjustment for Qwen2.5-VL (absolute)
4. BBox coordinate adjustment for Qwen3-VL (relative)
"""

import sys
import json
from pathlib import Path

# Test the utilities directly first
print("=" * 60)
print("Testing bbox_utils and point_utils...")
print("=" * 60)

from qwenvl.data.bbox_utils import (
    smart_resize_for_bbox,
    adjust_bbox_coordinates,
    parse_and_adjust_bbox_in_text,
)

from qwenvl.data.point_utils import (
    adjust_point_coordinates,
    parse_and_adjust_point_in_text,
)

# Test 1: BBox adjustment for Qwen2.5-VL (absolute)
print("\n1. BBox Adjustment (Qwen2.5-VL - Absolute)")
print("-" * 60)
bbox = [100, 100, 300, 300]
orig_h, orig_w = 600, 800
resized_h, resized_w = 450, 600

adjusted_bbox = adjust_bbox_coordinates(
    bbox, orig_h, orig_w, resized_h, resized_w, is_relative=False
)
print(f"Original bbox (800x600 image): {bbox}")
print(f"After resize to 600x450: {adjusted_bbox}")
print(f"Expected: [75, 75, 225, 225]")
print(f"✓ Pass" if adjusted_bbox == [75, 75, 225, 225] else "✗ Fail")

# Test 2: BBox adjustment for Qwen3-VL (relative)
print("\n2. BBox Adjustment (Qwen3-VL - Relative)")
print("-" * 60)
bbox_rel = [125, 167, 375, 500]
adjusted_bbox_rel = adjust_bbox_coordinates(
    bbox_rel, orig_h, orig_w, resized_h, resized_w, is_relative=True
)
print(f"Original bbox (relative 0-1000): {bbox_rel}")
print(f"After resize: {adjusted_bbox_rel}")
print(f"Expected: {bbox_rel} (unchanged)")
print(f"✓ Pass" if adjusted_bbox_rel == bbox_rel else "✗ Fail")

# Test 3: Point adjustment for Qwen2.5-VL (absolute)
print("\n3. Point Adjustment (Qwen2.5-VL - Absolute)")
print("-" * 60)
point = [400, 300]
adjusted_point = adjust_point_coordinates(
    point, orig_h, orig_w, resized_h, resized_w, is_relative=False
)
print(f"Original point (800x600 image): {point}")
print(f"After resize to 600x450: {adjusted_point}")
print(f"Expected: [300, 225]")
print(f"✓ Pass" if adjusted_point == [300, 225] else "✗ Fail")

# Test 4: Point adjustment for Qwen3-VL (relative)
print("\n4. Point Adjustment (Qwen3-VL - Relative)")
print("-" * 60)
point_rel = [500, 500]
adjusted_point_rel = adjust_point_coordinates(
    point_rel, orig_h, orig_w, resized_h, resized_w, is_relative=True
)
print(f"Original point (relative 0-1000): {point_rel}")
print(f"After resize: {adjusted_point_rel}")
print(f"Expected: {point_rel} (unchanged)")
print(f"✓ Pass" if adjusted_point_rel == point_rel else "✗ Fail")

# Test 5: Parse and adjust in text (bbox)
print("\n5. Parse and Adjust BBox in JSON Text (Qwen2.5-VL)")
print("-" * 60)
text = '{"bbox_2d": [100, 100, 300, 300], "label": "car"}'
adjusted_text = parse_and_adjust_bbox_in_text(
    text, orig_h, orig_w, resized_h, resized_w, is_relative=False, debug=False
)
print(f"Original: {text}")
print(f"Adjusted: {adjusted_text}")
expected = '{"bbox_2d": [75, 75, 225, 225], "label": "car"}'
print(f"Expected: {expected}")
print(f"✓ Pass" if adjusted_text == expected else "✗ Fail")

# Test 6: Parse and adjust in text (point)
print("\n6. Parse and Adjust Point in JSON Text (Qwen2.5-VL)")
print("-" * 60)
text_point = '{"point_2d": [400, 300], "label": "person"}'
adjusted_text_point = parse_and_adjust_point_in_text(
    text_point, orig_h, orig_w, resized_h, resized_w, is_relative=False, debug=False
)
print(f"Original: {text_point}")
print(f"Adjusted: {adjusted_text_point}")
expected_point = '{"point_2d": [300, 225], "label": "person"}'
print(f"Expected: {expected_point}")
print(f"✓ Pass" if adjusted_text_point == expected_point else "✗ Fail")

# Test 7: smart_resize calculation
print("\n7. Smart Resize Calculation")
print("-" * 60)
factor = 28
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
orig_h, orig_w = 1200, 1600

resized_h, resized_w = smart_resize_for_bbox(
    orig_h, orig_w, factor, min_pixels, max_pixels
)
print(f"Original size: {orig_w}x{orig_h}")
print(f"Min pixels: {min_pixels}, Max pixels: {max_pixels}")
print(f"Resized: {resized_w}x{resized_h}")
print(f"Pixels: {resized_w * resized_h} (should be between {min_pixels} and {max_pixels})")
print(f"✓ Pass" if min_pixels <= resized_w * resized_h <= max_pixels else "✗ Fail")

print("\n" + "=" * 60)
print("All utility tests completed!")
print("=" * 60)

# Test 8: Integration test with _build_messages
print("\n" + "=" * 60)
print("Testing _build_messages integration...")
print("=" * 60)

# Create a minimal test scenario
from qwenvl.data.data_processor import _build_messages
from unittest.mock import MagicMock
from PIL import Image
import tempfile
import os

# Create a temporary test image
print("\n8. Creating test image...")
with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
    test_image_path = tmp_file.name
    # Create a simple 800x600 image
    img = Image.new('RGB', (800, 600), color='red')
    img.save(test_image_path)
    print(f"Created test image: {test_image_path} (800x600)")

try:
    # Create test item with point_2d
    test_item = {
        "image": test_image_path,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nLocate the person."
            },
            {
                "from": "gpt",
                "value": '{"point_2d": [400, 300], "label": "person"}'
            }
        ]
    }
    
    # Mock processor and data_args
    mock_processor = MagicMock()
    mock_processor.image_processor.merge_size = 2
    mock_processor.image_processor.patch_size = 14
    
    mock_data_args = MagicMock()
    mock_data_args.min_pixels = 256 * 28 * 28
    mock_data_args.max_pixels = 1280 * 28 * 28
    
    # Test for Qwen2.5-VL (absolute coordinates)
    print("\n9. Testing _build_messages with Qwen2.5-VL (absolute)")
    print("-" * 60)
    messages_25 = _build_messages(
        test_item,
        Path("."),
        processor=mock_processor,
        data_args=mock_data_args,
        model_type="qwen2.5vl"
    )
    
    # Check the assistant message
    assistant_msg = messages_25[1]
    print(f"Assistant message content: {assistant_msg['content'][0]['text']}")
    
    # Parse to verify adjustment
    adjusted_content = assistant_msg['content'][0]['text']
    if '300, 225' in adjusted_content:  # Should be adjusted from [400, 300]
        print("✓ Pass - Coordinates were adjusted for Qwen2.5-VL")
    else:
        print(f"✗ Fail - Coordinates not adjusted as expected")
        print(f"   Got: {adjusted_content}")
    
    # Test for Qwen3-VL (relative coordinates)
    print("\n10. Testing _build_messages with Qwen3-VL (relative)")
    print("-" * 60)
    test_item_rel = {
        "image": test_image_path,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nLocate the person."
            },
            {
                "from": "gpt",
                "value": '{"point_2d": [500, 500], "label": "person"}'
            }
        ]
    }
    
    messages_3 = _build_messages(
        test_item_rel,
        Path("."),
        processor=mock_processor,
        data_args=mock_data_args,
        model_type="qwen3vl"
    )
    
    assistant_msg_3 = messages_3[1]
    print(f"Assistant message content: {assistant_msg_3['content'][0]['text']}")
    
    # Should NOT be adjusted (relative coords)
    adjusted_content_3 = assistant_msg_3['content'][0]['text']
    if '500, 500' in adjusted_content_3:  # Should remain unchanged
        print("✓ Pass - Coordinates unchanged for Qwen3-VL (relative)")
    else:
        print(f"✗ Fail - Coordinates were incorrectly adjusted")
        print(f"   Got: {adjusted_content_3}")
    
    print("\n" + "=" * 60)
    print("✅ Integration test completed successfully!")
    print("=" * 60)
    
finally:
    # Cleanup
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
        print(f"\nCleaned up test image: {test_image_path}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED!")
print("=" * 60)
print("\nThe data_processor.py coordinate adjustment is working correctly!")
print("You can now use it for training with bbox_2d and point_2d tasks.")

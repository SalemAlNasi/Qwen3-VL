#!/usr/bin/env python3
"""
Convert pointing dataset to Qwen format.

Handles both .json and .jsonl formats.
Extracts <ref>label</ref> and <point>[[x, y], ...]</point> from assistant responses
and converts to Qwen format with point_2d.

Usage:
    python convert_falcon_pointing_to_qwen.py input.jsonl output.json
    python convert_falcon_pointing_to_qwen.py input.json output.json
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any


def read_input_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read input file, handling both .json and .jsonl formats.
    
    Args:
        file_path: Path to input file
        
    Returns:
        List of data items
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            # Read line by line for jsonl
            return [json.loads(line.strip()) for line in f if line.strip()]
        else:
            # Read entire file for json
            return json.load(f)


def extract_label_from_ref(text: str) -> str:
    """
    Extract label from <ref>label</ref> tag.
    
    Args:
        text: Text containing <ref> tag
        
    Returns:
        Extracted label or "object" if not found
    """
    match = re.search(r'<ref>(.*?)</ref>', text)
    if match:
        return match.group(1).strip()
    return "object"


def extract_points_from_response(text: str) -> List[List[int]]:
    """
    Extract points from <point>[[x1, y1], [x2, y2], ...]</point> tag.
    Converts all coordinates to integers (rounding floats if necessary).
    
    Args:
        text: Assistant response text
        
    Returns:
        List of [x, y] coordinates as integers
    """
    # Find <point>...</point> content
    match = re.search(r'<point>(.*?)</point>', text, re.DOTALL)
    if not match:
        return []
    
    points_str = match.group(1).strip()
    
    # Parse the list of lists
    try:
        # Clean and parse
        points_str = points_str.replace('\n', '').replace(' ', '')
        points = json.loads(points_str)
        
        # Convert all coordinates to integers
        int_points = []
        for point in points:
            if isinstance(point, list) and len(point) == 2:
                # Convert each coordinate to int (round if float)
                x = int(round(float(point[0]))) if isinstance(point[0], (int, float)) else int(point[0])
                y = int(round(float(point[1]))) if isinstance(point[1], (int, float)) else int(point[1])
                int_points.append([x, y])
            else:
                print(f"Warning: Skipping invalid point format: {point}")
        
        return int_points
    except json.JSONDecodeError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: JSON parsing failed")
        print(f"Points string: {points_str[:200]}...")
        print(f"Error: {e}")
        print(f"{'='*60}\n")
        return []
    except (ValueError, TypeError) as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Point conversion failed")
        print(f"Points: {points}")
        print(f"Error: {e}")
        print(f"{'='*60}\n")
        return []


def convert_to_qwen_format(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single item to Qwen format.
    
    Args:
        item: Original data item
        
    Returns:
        Qwen format item or None if conversion fails
    """
    if "conversations" not in item or len(item["conversations"]) < 2:
        print(f"Warning: Skipping item {item.get('id', 'unknown')} - incomplete conversations")
        return None
    
    user_conv = item["conversations"][0]
    assistant_conv = item["conversations"][1]
    
    # Extract label from user or assistant message
    label = extract_label_from_ref(user_conv.get("value", ""))
    if label == "object":
        label = extract_label_from_ref(assistant_conv.get("value", ""))
    
    # Check if assistant says object is absent
    assistant_value = assistant_conv.get("value", "")
    if "absent" in assistant_value.lower() or "not" in assistant_value.lower():
        # Skip items where object is not present
        print(f"Info: Skipping item {item.get('id', 'unknown')} - object absent")
        return None
    
    # Extract points
    points = extract_points_from_response(assistant_value)
    
    if not points:
        print(f"\n{'='*60}")
        print(f"Warning: No points found in item {item.get('id', 'unknown')}")
        print(f"Image: {item.get('image', 'N/A')}")
        print(f"User message: {user_conv.get('value', 'N/A')[:100]}...")
        print(f"Assistant message: {assistant_value[:200]}...")
        print(f"{'='*60}\n")
        return None
    
    # Convert points to Qwen format
    qwen_points = []
    for point in points:
        if len(point) != 2:
            print(f"Warning: Invalid point format: {point}")
            continue
        qwen_points.append({
            "point_2d": point,
            "label": label
        })
    
    # Use standardized prompt template
    user_prompt = f"<image>\nLocate all {label} in this image and return points in JSON format."
    
    # Format output as JSON string
    qwen_response = json.dumps(qwen_points, ensure_ascii=False)
    
    return {
        "image": item.get("image", ""),
        "conversations": [
            {
                "from": "human",
                "value": user_prompt
            },
            {
                "from": "gpt",
                "value": qwen_response
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert pointing dataset to Qwen format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_falcon_pointing_to_qwen.py data.jsonl output.json
  python convert_falcon_pointing_to_qwen.py data.json output.json
  
Input format:
  {"id": 0, "image": "image.jpg", "conversations": [
    {"from": "user", "value": "Point at the <ref>person</ref> in <image>."},
    {"from": "assistant", "value": "<ref>person</ref> <point>[[100, 200], [300, 400]]</point>"}
  ]}
  
Output format:
  {"image": "image.jpg", "conversations": [
    {"from": "human", "value": "<image>\\nLocate all person in this image and return points in JSON format."},
    {"from": "gpt", "value": "[{\\"point_2d\\": [100, 200], \\"label\\": \\"person\\"}, {\\"point_2d\\": [300, 400], \\"label\\": \\"person\\"}]"}
  ]}
        """
    )
    parser.add_argument('input_file', help='Input file (.json or .jsonl)')
    parser.add_argument('output_file', help='Output file (.json)')
    parser.add_argument('--skip-absent', action='store_true', 
                       help='Skip items where object is absent (default: True)', 
                       default=True)
    parser.add_argument('--save-malformed', type=str,
                       help='Save malformed samples to this file (optional)',
                       default=None)
    
    args = parser.parse_args()
    
    print(f"Reading input from: {args.input_file}")
    
    # Read input
    try:
        input_data = read_input_file(args.input_file)
        print(f"Loaded {len(input_data)} items")
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Convert
    output_data = []
    malformed_data = []
    skipped = 0
    
    for i, item in enumerate(input_data):
        converted = convert_to_qwen_format(item)
        if converted is not None:
            output_data.append(converted)
        else:
            skipped += 1
            if args.save_malformed:
                malformed_data.append(item)
    
    print(f"\nConverted {len(output_data)} items successfully")
    print(f"Skipped {skipped} items")
    
    # Write output
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nOutput written to: {args.output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)
    
    # Save malformed samples if requested
    if args.save_malformed and malformed_data:
        try:
            with open(args.save_malformed, 'w', encoding='utf-8') as f:
                json.dump(malformed_data, f, indent=2, ensure_ascii=False)
            print(f"\nMalformed samples saved to: {args.save_malformed}")
        except Exception as e:
            print(f"Warning: Could not save malformed samples: {e}")
    
    # Print sample
    if output_data:
        print("\n" + "="*60)
        print("Sample converted item:")
        print("="*60)
        print(json.dumps(output_data[0], indent=2, ensure_ascii=False))
        print("="*60)


if __name__ == "__main__":
    main()

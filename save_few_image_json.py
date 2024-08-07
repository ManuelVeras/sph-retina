import json
import argparse
import logging
from decimal import Decimal
from sphdet.bbox.box_formator import deg2kent_single

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the filename of the JSON file
INPUT_FILENAME = 'datasets/360INDOOR/annotations/instances_val2017_transformed.json'
OUTPUT_FILENAME_TEMPLATE = 'datasets/annotations_small/instances_train2017_transformed_{}.json'

def transform_bbox(bbox):
    tensor_result = deg2kent_single(bbox)
    return tensor_result.tolist()  # Convert Tensor to list

def read_json_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {filename}")
        raise

def write_json_file(data, filename):
    try:
        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4)
    except IOError:
        logging.error(f"Error writing to file: {filename}")
        raise

def filter_annotations(annotations, image_ids, limit):
    filtered_annotations = [ann for ann in annotations if ann['image_id'] in image_ids]
    return filtered_annotations[:limit]

def transform_annotations(annotations):
    transformed_annotations = []
    for obj in annotations:
        obj['bbox'] = [float(value) if isinstance(value, Decimal) else value for value in obj['bbox']]
        obj['area'] = float(obj['area']) if isinstance(obj['area'], Decimal) else obj['area']
        obj['bbox'] = transform_bbox(obj['bbox'])
        transformed_annotations.append(obj)
    return transformed_annotations

def main(image_limit, object_limit):
    json_data = read_json_file(INPUT_FILENAME)
    annotations = json_data['annotations']
    images = json_data['images']

    # Filter images based on the image limit
    filtered_images = images[:image_limit]
    image_ids = [img['id'] for img in filtered_images]

    filtered_annotations = filter_annotations(annotations, image_ids, object_limit)
    #transformed_annotations = transform_annotations(filtered_annotations)

    # Update the JSON data with filtered images and annotations
    json_data['images'] = filtered_images
    json_data['annotations'] = filtered_annotations

    output_filename = OUTPUT_FILENAME_TEMPLATE.format(image_limit)
    write_json_file(json_data, output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform and save a subset of image annotations.')
    parser.add_argument('--image_limit', type=int, default=10, help='Number of images to process')
    parser.add_argument('--object_limit', type=int, default=1000, help='Number of objects to save')
    args = parser.parse_args()

    main(args.image_limit, args.object_limit)
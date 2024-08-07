import json
from decimal import Decimal
from sphdet.bbox.box_formator import deg2kent_single
import pdb

# Define the filename of the JSON file
INPUT_FILENAME = 'datasets/360INDOOR/annotations/instances_train2017.json'
OUTPUT_FILENAME_TEMPLATE = 'datasets/annotations_small/instances_train2017_transformed_5.json'

def transform_bbox(bbox):
    tensor_result = deg2kent_single(bbox)
    return tensor_result.tolist() # Convert Tensor to list


def main():
    # Read the entire JSON file
    with open(INPUT_FILENAME, 'r') as file:
        json_data = json.load(file)

    annotations = json_data['annotations']
    images = json_data['images']
    transformed_annotations = []

    # Limit the number of images to use
    IMAGE_LIMIT = 5  # Change this value to the desired number of images

    # Get the first few image IDs
    image_ids = [img['id'] for img in images[:IMAGE_LIMIT]]

    # Filter annotations to include only those with the selected image IDs
    filtered_annotations = [ann for ann in annotations if ann['image_id'] in image_ids]

    # Limit the number of objects to save
    LIMIT = 10  # Change this value to the desired number of objects

    for obj in filtered_annotations[:]:
        # Convert Decimal to float if needed
        obj['bbox'] = [float(value) if isinstance(value, Decimal) else value for value in obj['bbox']]
        obj['area'] = float(obj['area']) if isinstance(obj['area'], Decimal) else obj['area']
        
        # Transform the bbox values
        obj['bbox'] = transform_bbox(obj['bbox'])
        
        # Add the transformed object to the list
        transformed_annotations.append(obj)

    # Update the annotations in the original JSON structure
    json_data['annotations'] = transformed_annotations

    # Extract object names for the filename
    object_names = [obj.get('name', 'unknown') for obj in transformed_annotations]
    object_names_str = '_'.join(object_names[:LIMIT])

    # Create the output filename with object names
    output_filename = OUTPUT_FILENAME_TEMPLATE.format(object_names_str)

    # Write the updated JSON structure to a new file
    with open(output_filename, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)


if __name__ == "__main__":
    main()
# path/filename: read_large_json.py
# Import necessary libraries
import ijson
import json
from decimal import Decimal
import pdb
from kent.bfov2kent import deg2kent

# Define the filename of the JSON file
input_filename = 'datasets/360INDOOR/annotations/instances_val2017.json'
output_filename = 'datasets/360INDOOR/annotations/instances_val2017_transformed.json'


#TODO: Define the transformation function
def transform_bbox(bbox):
    # Example transformation: scale the bbox values by a factor of 2
    return deg2kent(bbox)

# Read the entire JSON file
with open(input_filename, 'r') as file:
    json_data = json.load(file)

# Process the annotations
annotations = json_data['annotations']
transformed_annotations = []

for obj in annotations:
    # Convert Decimal to float if needed
    obj['bbox'] = [float(value) if isinstance(value, Decimal) else value for value in obj['bbox']]
    obj['area'] = float(obj['area']) if isinstance(obj['area'], Decimal) else obj['area']
    
    # Transform the bbox values
    obj['bbox'] = transform_bbox(obj['bbox'])
    
    # Add the transformed object to the list
    transformed_annotations.append(obj)

# Update the annotations in the original JSON structure
json_data['annotations'] = transformed_annotations

# Write the updated JSON structure to a new file
with open(output_filename, 'w') as outfile:
    json.dump(json_data, outfile, indent=4)

# For debugging, use pdb
#import pdb; pdb.set_trace()

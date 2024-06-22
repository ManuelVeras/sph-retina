# path/filename: read_large_json.py
# Import necessary libraries
import ijson
import json
from decimal import Decimal

# Define the filename of the JSON file
input_filename = 'datasets/360INDOOR/annotations/instances_val2017.json'
output_filename = 'datasets/360INDOOR/annotations/instances_val2017_transformed.json'


#TODO: Define the transformation function
def transform_bbox(bbox):
    # Example transformation: scale the bbox values by a factor of 2
    return [value * 2 for value in bbox]

# Open the input JSON file and iterate over items
with open(input_filename, 'r') as file:
    # Assuming the JSON structure is an array of objects
    objects = ijson.items(file, 'annotations.item')
    
    # Prepare a list to hold the transformed objects
    transformed_objects = []

    for obj in objects:
        # Convert Decimal to float if needed
        obj['bbox'] = [float(value) if isinstance(value, Decimal) else value for value in obj['bbox']]
        obj['area'] = float(obj['area']) if isinstance(obj['area'], Decimal) else obj['area']
        
        # Transform the bbox values
        obj['bbox'] = transform_bbox(obj['bbox'])
        
        # Add the transformed object to the list
        transformed_objects.append(obj)

# Write the transformed objects to a new JSON file
with open(output_filename, 'w') as outfile:
    json.dump({'annotations': transformed_objects}, outfile, indent=4)

# For debugging, use pdb
import pdb; pdb.set_trace()

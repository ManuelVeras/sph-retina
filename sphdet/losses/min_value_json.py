import json

# Load the JSON file
with open('datasets/360INDOOR/annotations/instances_val2017_transformed.json') as f:
    data = json.load(f)

# Initialize a variable to store the minimum value
min_value = float('inf')

# Iterate through all annotations
for annotation in data:
    bbox = annotation.get('bbox', [])
    if len(bbox) >= 4:
        min_value = min(min_value, bbox[3])

print(f"The minimum value of the 4th item in bbox is: {min_value}")
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={
        "image": np.zeros((256, 256, 3), dtype=np.uint8),  # replace with your real image
        "instruction": "pick up the red block",
        "unnorm_key": "bridge_orig"
    }
).json()

print(action)

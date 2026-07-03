import json

import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.get(
    "http://0.0.0.0:8000/profiler/full",
).json()

print(json.dumps(action, indent=2))

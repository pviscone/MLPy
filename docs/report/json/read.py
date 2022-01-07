import json
import numpy

with open(pv.json) as json_file:
    f_dict = json.load(json_file)
    
for i in range(0,10): print(f_dict[i])

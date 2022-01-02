import numpy
import json

#loading dictionary grid_search
with open('grid_search_pv.json') as json_file:
    grid_results0 = json.load(json_file)

with open('grid_search_pv2.json') as json_file:
    grid_results1 = json.load(json_file)

with open('lrelu.json') as json_file:
    grid_results2 = json.load(json_file)
    
with open('grid_search_mix.json') as json_file:
    grid_results3 = json.load(json_file)

 with open('') as json_file:
    grid_results4 = json.load(json_file)
    
for i in range(0,10):
	print(grid_results[i],'\n')

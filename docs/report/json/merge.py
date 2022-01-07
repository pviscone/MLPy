import numpy as np
import json


file_name = __file__.split('/')[-1]
final_dir = __file__[:-len(file_name)]

mypath = final_dir

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.append('all_dict.json')
list_exclude = ['all_dict.json']
json_files = [f for f in onlyfiles if f.endswith('.json') and f not in list_exclude]

all_dict = []
for f in json_files:
    print(f)
    with open(mypath + f) as json_file:
        f_dict = json.load(json_file)
    for d in f_dict:
        all_dict.append(d)

err = []
for d in all_dict:
    err.append(d['Error'])

arr_err = np.array(err)
sort_err_idx = np.argsort(arr_err)

final_dict_list = []
for i in range(len(all_dict)):
    idx = sort_err_idx[i]
    final_dict_list.append(all_dict[idx])

with open(mypath + 'all_dict.json','w') as fp:
    json.dump(final_dict_list,fp)



print('AND THE WINNER IS.......')
first_n = 10
for i in range(first_n):
    print('-', final_dict_list[i]['model']['structure'])
    print('-', final_dict_list[i]['model']['func'])
    print('error:', final_dict_list[i]['Error'])

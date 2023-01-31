import json
import copy 

multi = True

if multi:
    source_list = ["photo", "cartoon", "art_painting", "sketch"]
    new_dict = {}
    for cur_domain in source_list:
        link_path = f'./link/{cur_domain}_test_link.json'
        link_dict = json.load(open(link_path))
        new_dict.update(link_dict)
    outfile = './link/pacs_minimal_link.json'
    with open(outfile,'w') as f:
        json.dump(new_dict,f) 

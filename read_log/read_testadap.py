import pandas as pd
import numpy as np


source_list = ['art_painting','photo','sketch','cartoon']
algo_list = ['ERM','OURS']

base_dir = '/homes/55/jianhaoy/projects/EKI/results/Test_adap'



single = True

# csv_out = '/homes/55/jianhaoy/projects/EKI/results/raw_data'
if single:
    log_csv_lis = []
    for domain in source_list:
        print('-'*60)
        print(f'Source Domain:{domain}')
        cols = ['Algorithm']
        cols.append('test')
        cols.append('adaptation')
        for i in source_list:
            if i != domain:
                cols.append(i)
                cols.append(f'adap_{i}')
        log_csv = pd.DataFrame(columns=cols)
        log_csv['Algorithm'] = algo_list
        log_csv.set_index('Algorithm', inplace = True)
        for algo in algo_list:
            # print(algo)
            log_path = f'{base_dir}/{algo}/{domain}.out'
            f = open(log_path,'r').readlines()[-18:-3]
            # print(f)
            for i in f:
                if '---' in i:
                    continue
                toks = i.split(':')
                test_domain = toks[0].split('>')[0].split(' ')[-1]
                test_acc = toks[1].strip('/\n')
                # print(acc)

                test_acc = float(test_acc)
                test_acc = str(round(test_acc, 2))
                # # print(algo,domain,'----->',test_domain,':',test_acc)
                # # print(test_domain,algo)
                log_csv[test_domain][algo] = test_acc
        print(log_csv)
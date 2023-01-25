import pandas as pd
import numpy as np


source_list = ['art_painting','photo','sketch','cartoon']
algo_list = ['ERM','OURS']
base_dir = '/homes/55/jianhaoy/projects/NJP/results/Trail1'



log_csv_lis = []
for domain in source_list:
    print('-'*60)
    print(f'Source Domain:{domain}')
    cols = ['Algorithm']
    for i in source_list:
        if i != domain:
            cols.append(i)
    cols.append('Average')
    log_csv = pd.DataFrame(columns=cols)
    log_csv['Algorithm'] = algo_list
    log_csv.set_index('Algorithm', inplace = True)
    for algo in algo_list:
        # print(algo)
        log_path = f'{base_dir}/{algo}/{domain}.out'
        f = open(log_path,'r').readlines()[-10:]
        for i in f:
            if '*' in i:
                continue
            toks = i.split(' ')
            if len(toks) == 1:
                continue
            if len(toks) == 10:
                test_domain = 'Average'
                test_acc = toks[-5]
            else:
                test_domain = toks[-3]
                test_acc = toks[-1].strip('\n')

            test_acc = float(test_acc) * 100
            test_acc = str(round(test_acc, 2))
            # print(algo,domain,'----->',test_domain,':',test_acc)
            # print(test_domain,algo)
            log_csv[test_domain][algo] = test_acc
    print(log_csv)
    log_csv_lis.append(log_csv)

# PRINT LINE
rows = list(log_csv.index)

line = 'Source Domain '
for i in source_list:
    line += '& \multicolumn{3}{c}' + '{'+ f'{i}' +'}'
line += ' \\\\'
print(line)
line = 'Target Domain '
for log_csv in log_csv_lis:
    for dom in list(log_csv.columns):
        if dom == 'Average':
            continue
        if dom == 'art_painting':
            dom = 'art'
        line += f'& {dom} '
line += ' \\\\'
print(line)
print('\\cline{1-13}')
for alg in rows:
    if (alg == 'OURS_G'):
        print('\\cline{1-13}')
    line = f'{alg} '
    for log_csv in log_csv_lis:
        for dom in list(log_csv.columns):
            if dom == 'Average':
                continue
            line += f'& ${log_csv[dom][alg]}$ '
    line += ' \\\\'
    print(line)

# Average
line = 'Source Domain '
print(line)
for log_csv in log_csv_lis:
    for dom in list(log_csv.columns):
        if dom != 'Average':
            continue
        line += f'& {dom} '
line += ' \\\\'
print(line)
print('\\cline{1-13}')
for alg in rows:
    if (alg == 'OURS_G'):
        print('\\cline{1-13}')
    line = f'{alg} '
    for log_csv in log_csv_lis:
        for dom in list(log_csv.columns):
            if dom != 'Average':
                continue
            line += f'& ${log_csv[dom][alg]}$ '
    line += ' \\\\'
    print(line)
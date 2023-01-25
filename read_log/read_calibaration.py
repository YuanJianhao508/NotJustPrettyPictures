import pandas as pd
import numpy as np
import json
import copy


source_list = ['art_painting','photo','sketch','cartoon']
# source_list = ['Art','Clipart','Product','Real_World']

algo_list = ['ERM','MixUp','CutMix','AugMix','RandAugment','SN','SRN','CutOut','RSC','MEADA','L2D','PixMix','ACVC','OURS']



base_dir = '/homes/55/jianhaoy/projects/EKI/results/ForTest'

single = True


if single:
    cols = ['Algorithm']
    for i in source_list:
        cols.append(i)
    cols.append('Average')
    log_csv = pd.DataFrame(columns=cols)
    log_csv['Algorithm'] = algo_list
    log_csv.set_index('Algorithm', inplace = True)
    rej_csv = log_csv.copy(deep=True)
    for domain in source_list:
        for algo in algo_list:
            print(algo)
            log_path = f'{base_dir}/{algo}/{domain}.out'
            f = open(log_path,'r').readlines()[-1:]
            # print(f)
            for k in f:
                dic = json.loads(k.strip('\\n').replace("\'", "\""))
                ada = dic['AdaECE']
                rej = dic['Rejection']
                for test_acc in [ada,rej]:
                    test_acc = float(test_acc) * 100
                    test_acc = str(round(test_acc, 2))
                # print(algo,domain,'----->',test_domain,':',test_acc)
                # print(test_domain,algo)
                log_csv[domain][algo] = ada
                rej_csv[domain][algo] = rej

    
    for algo in algo_list:
        s = 0
        for domain in source_list:
            s += float(log_csv[domain][algo])
        log_csv['Average'][algo] = str(round(s/4, 2))
            
    print(log_csv)
    log_csv_lis = [log_csv]
    # # PRINT LINE
    rows = list(log_csv.index)

    line = 'Source Domain '
    for i in source_list:
        line += '& \multicolumn{3}{c}' + '{'+ f'{i}' +'}'
    line += ' \\\\'
    print(line)
    line = 'Target Domain '
    for log_csv in log_csv_lis:
        for dom in list(log_csv.columns):
            # if dom == 'Average':
            #     continue
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
                # if dom == 'Average':
                #     continue
                line += f'& ${str(round(float(log_csv[dom][alg]),2))}$ '
        line += ' \\\\'
        print(line)


    # Rejection
    log_csv = rej_csv
    for algo in algo_list:
        s = 0
        for domain in source_list:
            s += float(log_csv[domain][algo])
        log_csv['Average'][algo] = str(round(s/4, 2))
                
    print(log_csv)
    log_csv_lis = [log_csv]
    # # PRINT LINE
    rows = list(log_csv.index)

    line = 'Source Domain '
    for i in source_list:
        line += '& \multicolumn{3}{c}' + '{'+ f'{i}' +'}'
    line += ' \\\\'
    print(line)
    line = 'Target Domain '
    for log_csv in log_csv_lis:
        for dom in list(log_csv.columns):
            # if dom == 'Average':
            #     continue
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
                # if dom == 'Average':
                #     continue
                line += f'& ${str(round(float(log_csv[dom][alg]),2))}$ '
        line += ' \\\\'
        print(line)


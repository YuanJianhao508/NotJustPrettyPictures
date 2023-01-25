import pandas as pd
import numpy as np


# source_list = ['art_painting','photo','sketch','cartoon']
source_list = ['Art','Clipart','Product','Real_World']
algo_list = ['H']

# algo_list = ['ERM','MixUp','CutMix','AugMix','RandAugment','CutOut','RSC','SN','SRN','MEADA','L2D','PixMix','ACVC','OURS_G','OURS_SD','Con','Mod','hd','Textinv','CLIPH']
# algo_list = ['OURS','OURS_Multi']
# algo_list = ['dumb','engi1','text_inv','language_enhancement_conservative','language_enhancement_moderate']
# algo_list = ['OURS']
# algo_list = ['ERM',0.25,0.5,0.75,1]
base_dir = '/homes/55/jianhaoy/projects/EKI/results_Officehome/untar'
# base_dir = '/homes/55/jianhaoy/projects/EKI/results/untar_engi'



single = True

if single:
    line = []
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
        # print(log_csv)
        s = 0
        for red in [i for i in source_list if i != domain]:

            log_path = f'{base_dir}/{red}/{domain}.out'
            f = open(log_path,'r').readlines()[-10:]
            for i in f:
                # print(i)
                if '*' in i:
                    continue
                toks = i.split(' ')
                if len(toks) == 1:
                    continue
                if len(toks) == 10:
                    test_domain = 'Average'
                    test_acc = toks[-5]
                    test_acc = float(test_acc) * 100
                    test_acc = str(round(test_acc, 2))
                else:
                    test_domain = toks[-3]
                    # print(test_domain)
                    if test_domain != red:
                        continue
                    test_acc = toks[-1].strip('\n')

                    test_acc = float(test_acc) * 100
                    s += test_acc
                    # print(s)
                    test_acc = str(round(test_acc, 2))
                # print(algo,domain,'----->',test_domain,':',test_acc)
                # print(test_domain,algo)
                log_csv[red]['H'] = test_acc
            
        # print(s)
        log_csv['Average'] = str(round(s/3,2))
        print(log_csv.loc['H'][:3].tolist())
        line.append(log_csv.loc['H'][:3].tolist())
    
    lines = [item for sublist in line for item in sublist]
    res = [f'& ${i}$' for i in lines]
    res = ' '.join(res)
    print(res)


    # # PRINT LINE
    # rows = list(log_csv.index)

    # line = 'Source Domain '
    # for i in source_list:
    #     line += '& \multicolumn{3}{c}' + '{'+ f'{i}' +'}'
    # line += ' \\\\'
    # print(line)
    # line = 'Target Domain '
    # for log_csv in log_csv_lis:
    #     for dom in list(log_csv.columns):
    #         if dom == 'Average':
    #             continue
    #         if dom == 'art_painting':
    #             dom = 'art'
    #         line += f'& {dom} '
    # line += ' \\\\'
    # print(line)
    # print('\\cline{1-13}')
    # for alg in rows:
    #     if (alg == 'OURS_G'):
    #         print('\\cline{1-13}')
    #     line = f'{alg} '
    #     for log_csv in log_csv_lis:
    #         for dom in list(log_csv.columns):
    #             if dom == 'Average':
    #                 continue
    #             line += f'& ${log_csv[dom][alg]}$ '
    #     line += ' \\\\'
    #     print(line)
    
    # # Average
    # line = 'Source Domain '
    # print(line)
    # for log_csv in log_csv_lis:
    #     for dom in list(log_csv.columns):
    #         if dom != 'Average':
    #             continue
    #         line += f'& {dom} '
    # line += ' \\\\'
    # print(line)
    # print('\\cline{1-13}')
    # for alg in rows:
    #     if (alg == 'OURS_G'):
    #         print('\\cline{1-13}')
    #     line = f'{alg} '
    #     for log_csv in log_csv_lis:
    #         for dom in list(log_csv.columns):
    #             if dom != 'Average':
    #                 continue
    #             line += f'& ${log_csv[dom][alg]}$ '
    #     line += ' \\\\'
    #     print(line)


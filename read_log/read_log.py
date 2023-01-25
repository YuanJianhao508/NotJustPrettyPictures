import pandas as pd
import numpy as np


source_list = ['art_painting','photo','sketch','cartoon']
# source_list = ['Art','Clipart','Product','Real_World']

# algo_list = ['ERM','MixUp','CutMix','AugMix','RandAugment','CutOut','RSC','SN','SRN','MEADA','L2D','PixMix','ACVC','OURS_G','OURS_SD','Con','Mod','hd','Textinv','CLIPH','L2']
# algo_list = ["AugMix","RandAugment","CutOut","MixUp","CutMix", "RSC"]
# algo_list = ['OURS','OURS_Multi']
# algo_list = ['dumb','engi1','text_inv','language_enhancement_conservative','language_enhancement_moderate']
algo_list = ['ERM','MixUp','CutMix','AugMix','RandAugment','CutOut','RSC','MEADA', 'ACVC', 'PixMix', 'L2D' ,'OURS']
# algo_list = ['ERM',0.25,0.5,0.75,1]
base_dir = '/homes/55/jianhaoy/projects/EKI/results/Rebuttal'
# trails = [f'/homes/55/jianhaoy/projects/EKI/results/E30_PRET_CE{i}' for i in [1,2,3,4,5]]

# base_dir = '/homes/55/jianhaoy/projects/EKI/results/text_compair'
# base_dir = '/homes/55/jianhaoy/projects/EKI/results/E30_PRET_CE4'
# base_dir = '/homes/55/jianhaoy/projects/EKI/results/text_compair'
# base_dir= '/homes/55/jianhaoy/projects/EKI/results/Moderate'
# trails = [f'/homes/55/jianhaoy/projects/EKI/results/Conservative/{i}' for i in [1,2,3,4,5]]
single = True
# print(trail_lis)
csv_out = '/homes/55/jianhaoy/projects/EKI/results/raw_data'
if single:
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

else:
    trail_res_lis = []
    for trail in trails:
        log_csv_lis = []
        for domain in source_list:
            # print('-'*60)
            # print(f'Source Domain:{domain}')
            cols = ['Algorithm']
            for i in source_list:
                if i != domain:
                    cols.append(i)
            cols.append('Average')
            log_csv = pd.DataFrame(columns=cols)
            log_csv['Algorithm'] = algo_list
            log_csv.set_index('Algorithm', inplace = True)
            for algo in algo_list:
                log_path = f'{trail}/{algo}/{domain}.out'
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
            # print(log_csv)
            log_csv_lis.append(log_csv)
        trail_res_lis.append(log_csv_lis)

    sdf_lis = []
    for idom in range(len(log_csv_lis)):
        print(f'Source Domain: {source_list[idom]}')
        dom_set = []
        for itrail in range(len(trail_res_lis)):

            dom_set.append(trail_res_lis[itrail][idom])
        
        cols = list(dom_set[0].columns)
        rows = list(log_csv.index)
        sdf = dom_set[0]
        for row in rows:
            for col in cols:
                num_lis = []
                for df in dom_set:
                    # print(df[col][row],col,row)
                    num_lis.append(float(df[col][row]))
                s = np.array(num_lis)
                mu,std = np.mean(s),np.std(s)
                mu,std = str(round(mu, 2)),str(round(std, 2))
                # print(mu,std,col,row)
                # sdf[col][row] = f'{mu} \u00B1 {std}'
                sdf[col][row] = mu
        print(sdf)
        # sdf.to_csv(f'{csv_out}/Source-{source_list[idom]}.csv')
        sdf_lis.append(sdf)


    # PRINT LINE
    rows = list(sdf.index)

    line = 'Source Domain '
    for i in source_list:
        line += '& \multicolumn{3}{c}' + '{'+ f'{i}' +'}'
    line += ' \\\\'
    print(line)
    line = 'Target Domain '
    for log_csv in sdf_lis:
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
        for log_csv in sdf_lis:
            for dom in list(log_csv.columns):
                if dom == 'Average':
                    continue
                line += f'& ${log_csv[dom][alg]}$ '
        line += ' \\\\'
        print(line)

    # Average
    rows = list(sdf.index)

    line = 'Source Domain '
    for i in source_list:
        line += '& \multicolumn{3}{c}' + '{'+ f'{i}' +'}'
    line += ' \\\\'
    print(line)
    line = 'Target Domain '
    for log_csv in sdf_lis:
        for dom in list(log_csv.columns):
            if dom != 'Average':
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
        flis = []
        for log_csv in sdf_lis:
            for dom in list(log_csv.columns):
                if dom != 'Average':
                    continue
                line += f'& ${log_csv[dom][alg]}$ '
                flis.append(float(log_csv[dom][alg]))
        
        aav = sum(flis)/len(flis)
        # print(flis)
        # print(aav)
        line += f'& ${str(round(aav, 2))}$ '
        line += ' \\\\'
        print(line)

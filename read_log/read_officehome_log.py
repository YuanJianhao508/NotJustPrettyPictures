import pandas as pd
import numpy as np


source_list = ['Art','Clipart','Product','Real_World']
algo_list = ['ERM','MixUp','CutMix','AugMix','RandAugment','CutOut','RSC','SN','SRN','MEADA','L2D','PixMix','ACVC','OURS_G','OURS_SD','Con','Mod','Min','MinG']
# algo_list = ['OURS']
# trail_lis = [f'trail_{i}' for i in range(1,6)]
trails = [f'/homes/55/jianhaoy/projects/EKI/results_Officehome/E50_PRET_CE{i}' for i in [1,2,3,4,5]]
single = False
csv_out = '/homes/55/jianhaoy/projects/EKI/results_Officehome/raw_data'
# print(trail_lis)

if single:
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
            log_path = f'/homes/55/jianhaoy/projects/CIRL/results_Officehome/E180_finetune/{algo}/{domain}.out'
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
                log_csv[test_domain][algo] = test_acc
        print(log_csv)
        # log_csv.to_csv('log.csv')
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
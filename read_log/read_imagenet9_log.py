import pandas as pd
import numpy as np


source_list = ['original','mixed_rand','mixed_same']
# source_list = ['original']

# algo_list = ['ERM','MixUp','CutMix','AugMix','RandAugment','CutOut','RSC','SN','SRN','MEADA','PixMix','ACVC','L2D','OURS_G','OURS_D']
# algo_list = ['ERM','OURS_D','Inpaint']
algo_list = ['MixUp','CutMix']

# trails = [f'/homes/55/jianhaoy/projects/EKI/results_ImageNet9/E30_PRET_CE{i}' for i in [1,2,3,4,5]]
trails = [f'/homes/55/jianhaoy/projects/EKI/results_ImageNet9/Crossval/{i}' for i in [1,2,3,4,5]]

# base_dir = '/homes/55/jianhaoy/projects/EKI/results_ImageNet9/E30_PRET_CE1'
base_dir = '/homes/55/jianhaoy/projects/EKI/results_ImageNet9/Crossval/1'


single = False

# print(trail_lis)

if single:
    for domain in ['original']:
        print('-'*60)
        print(f'Source Domain:{domain}')
        cols = ['Algorithm']
        cols.append('In')
        for i in source_list:
            if i != domain:
                cols.append(i)
        
        log_csv = pd.DataFrame(columns=cols)
        log_csv['Algorithm'] = algo_list
        log_csv.set_index('Algorithm', inplace = True)
        # print(log_csv)
        for algo in algo_list:
            log_path = f'{base_dir}/{algo}/{domain}.out'
            f = open(log_path,'r').readlines()[-7:]
            # print(algo)
            for i in f:
                if '*' in i:
                    continue
                toks = i.split(' ')
                if len(toks) == 1:
                    continue
                if len(toks) == 10:
                    test_domain = 'In'
                    test_acc = toks[2].strip(',')
                else:
                    test_domain = toks[-3]
                    test_acc = toks[-1].strip('\n')

                test_acc = float(test_acc) * 100
                test_acc = str(round(test_acc, 2))
                # print(algo,domain,'----->',test_domain,':',test_acc)
                # print(test_domain,algo)
                log_csv[test_domain][algo] = test_acc
        # PRINT CSV
        gap_res = []
        rows = list(log_csv.index)
        for alg in rows:
            line = f'{alg} '
            gap_lis = []
            for dom in list(log_csv.columns):
                if dom == 'Average':
                    continue
                gap_lis.append(float(log_csv[dom][alg]))
            gap_res.append(str(round(gap_lis[2]-gap_lis[1],2)))
        log_csv['Gap'] = gap_res
        # log_csv.drop(['Average'], axis=1, inplace=True)
        print(log_csv)


        # PRINT LINE
        rows = list(log_csv.index)
        for alg in rows:
            line = f'{alg} '
            gap_lis = []
            for dom in list(log_csv.columns):
                if dom == 'Average':
                    continue
                gap_lis.append(float(log_csv[dom][alg]))
                line += f'& ${log_csv[dom][alg]}$ '
            line += f'& ${str(round(gap_lis[1]-gap_lis[0],2))}$ '
            line += ' \\\\'
            print(line)

else:
    trail_res_lis = []
    for trail in trails:
        log_csv_lis = []
        for domain in ['original']:
            cols = ['Algorithm']
            cols.append('In')
            for i in source_list:
                if i != domain:
                    cols.append(i)
            
            log_csv = pd.DataFrame(columns=cols)
            log_csv['Algorithm'] = algo_list
            log_csv.set_index('Algorithm', inplace = True)
            # print(log_csv)
            for algo in algo_list:
                log_path = f'{trail}/{algo}/{domain}.out'
                f = open(log_path,'r').readlines()[-7:]
                # print(algo)
                for i in f:
                    if '*' in i:
                        continue
                    toks = i.split(' ')
                    # print(toks,len(toks))
                    if len(toks) == 1:
                        continue
                    if len(toks) == 10:
                        test_domain = 'In'
                        test_acc = toks[2].strip(',')
                    else:
                        test_domain = toks[-3]
                        test_acc = toks[-1].strip('\n')

                    test_acc = float(test_acc) * 100
                    test_acc = str(round(test_acc, 2))
                    # print(algo,domain,'----->',test_domain,':',test_acc)
                    # print(test_domain,algo)
                    log_csv[test_domain][algo] = test_acc

            # PRINT CSV
            gap_res = []
            rows = list(log_csv.index)
            for alg in rows:
                line = f'{alg} '
                gap_lis = []
                for dom in list(log_csv.columns):
                    # print(dom)
                    if dom == 'Average':
                        continue
                    gap_lis.append(float(log_csv[dom][alg]))
                gap_res.append(str(round(gap_lis[2]-gap_lis[1],2)))
            log_csv['Gap'] = gap_res
            # log_csv.drop(['Average'], axis=1, inplace=True)
            log_csv_lis.append(log_csv)
        trail_res_lis.append(log_csv_lis)
    # print(trail_res_lis)
    sdf_lis = []
    for idom in range(1):
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
        sdf_lis.append(sdf)


    # # PRINT LINE
    rows = list(sdf.index)
    for alg in rows:
        line = f'{alg} '
        gap_lis = []
        for dom in list(sdf.columns):
            cont = str(sdf[dom][alg])
            if str(cont) == 4:
                cont = str(cont)+'0'

            line += f'& ${cont}$ '
        line += ' \\\\'
        print(line)

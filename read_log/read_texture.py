import pandas as pd
import numpy as np


source_list = ['original','in','random','edge','silo']
# source_list = ['original']

# algo_list = ['ERM','MixUp','CutMix','AugMix','RandAugment','SN','SRN','CutOut','RSC','MEADA','PixMix','ACVC','L2D','OURS']

# algo_list = ["AugMix","RandAugment","CutOut","MixUp","CutMix", "RSC"]
algo_list = ['MixUp','CutMix']


# trails = [f'/homes/55/jianhaoy/projects/EKI/results_Texture/pg/E30_PRET_CE{i}' for i in [1,2,3,4,5]]
trails = [f'/homes/55/jianhaoy/projects/EKI/results_Texture/Crossval/{i}' for i in [1,2,3,4,5]]

# base_dir = '/homes/55/jianhaoy/projects/EKI/results_Texture/Crossval/1'
# base_dir = '/homes/55/jianhaoy/projects/EKI/results_ImageNet9/test1'

single = False

# print(trail_lis)

if single:
    for domain in ['original']:
        print('-'*60)
        print(f'Source Domain:{domain}')
        cols = ['Algorithm']
        # cols.append('In')
        for i in source_list:
            if i != domain:
                cols.append(i)
        
        log_csv = pd.DataFrame(columns=cols)
        log_csv['Algorithm'] = algo_list
        log_csv.set_index('Algorithm', inplace = True)
        # print(log_csv)
        for algo in algo_list:
            log_path = f'{base_dir}/{algo}/{domain}.out'
            f = open(log_path,'r').readlines()[-10:]
            print(algo)
            print(f)
            for i in f:
                # print(i)
                if '*' in i:
                    continue
                toks = i.split(' ')
                if len(toks) == 1:
                    continue
                # if len(toks) == 10:
                #     test_domain = 'In'
                #     test_acc = toks[2].strip(',')
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
        cols = list(log_csv.columns)
        for pair in [(0,1)]:
            a,b = pair[0],pair[1]
            print(a,b)
            for alg in rows:
                line = f'{alg} '
                gap_lis = []
                for dom in list(log_csv.columns):
                    if dom == 'Average':
                        continue
                    gap_lis.append(float(log_csv[dom][alg]))
                gap_res.append(str(round(gap_lis[a]-gap_lis[b],2)))
            log_csv[f'{cols[a]}-{cols[b]}'] = gap_res
        # log_csv.drop(['Average'], axis=1, inplace=True)
        # print(log_csv)
        acc_log = log_csv

       # CCS 
        ccs_dir = '/homes/55/jianhaoy/projects/EKI/results_Texture/ForTest'
        cols = ['Algorithm']
        for i in ['CCS']:
            cols.append(i)
        # cols.append('Average')
        log_csv = pd.DataFrame(columns=cols)
        log_csv['Algorithm'] = algo_list
        log_csv.set_index('Algorithm', inplace = True)
        for domain in ['original']:
            for algo in algo_list:
                # print(algo)
                log_path = f'{ccs_dir}/{algo}/{domain}.out'
                f = open(log_path,'r').readlines()[-1:]
                # print(f)
                for i in f:
                    # print(i)
                    test_acc = i
                    # print(toks)
                    test_domain = 'CCS'
                    test_acc = float(test_acc) * 100
                    test_acc = str(round(test_acc, 2))
                    # print(algo,domain,'----->',test_domain,':',test_acc)
                    # print(test_domain,algo)
                    log_csv[test_domain][algo] = test_acc
        # print(log_csv)
        ccs_csv = log_csv

        acc_log['CCS'] = ccs_csv['CCS']
        print(acc_log)

        # PRINT LINE
        # rows = list(log_csv.index)
        # for alg in rows:
        #     line = f'{alg} '
        #     gap_lis = []
        #     for dom in list(log_csv.columns):
        #         if dom == 'Average':
        #             continue
        #         gap_lis.append(float(log_csv[dom][alg]))
        #         line += f'& ${log_csv[dom][alg]}$ '
        #     # line += f'& ${str(round(gap_lis[1]-gap_lis[0],2))}$ '
        #     line += ' \\\\'
        #     print(line)

else:
    trail_res_lis = []
    for trail in trails:
        log_csv_lis = []
        for domain in ['original']:
            # print('-'*60)
            # print(f'Source Domain:{domain}')
            cols = ['Algorithm']
            # cols.append('In')
            for i in source_list:
                if i != domain:
                    cols.append(i)
            
            log_csv = pd.DataFrame(columns=cols)
            log_csv['Algorithm'] = algo_list
            log_csv.set_index('Algorithm', inplace = True)
            # print(log_csv)
            for algo in algo_list:
                log_path = f'{trail}/{algo}/{domain}.out'
                f = open(log_path,'r').readlines()[-10:]
                # print(algo)
                # print(f)
                for i in f:
                    # print(i)
                    if '*' in i:
                        continue
                    toks = i.split(' ')
                    if len(toks) == 1:
                        continue
                    # if len(toks) == 10:
                    #     test_domain = 'In'
                    #     test_acc = toks[2].strip(',')
                    else:
                        test_domain = toks[-3]
                        test_acc = toks[-1].strip('\n')

                    test_acc = float(test_acc) * 100
                    test_acc = str(round(test_acc, 2))
                    # print(algo,domain,'----->',test_domain,':',test_acc)
                    # print(test_domain,algo)
                    log_csv[test_domain][algo] = test_acc
                    log_csv_lis.append(log_csv)

        # PRINT CSV
        gap_res = []
        rows = list(log_csv.index)
        cols = list(log_csv.columns)
        for pair in [(0,1)]:
            a,b = pair[0],pair[1]
            # print(a,b)
            for alg in rows:
                line = f'{alg} '
                gap_lis = []
                for dom in list(log_csv.columns):
                    if dom == 'Average':
                        continue
                    gap_lis.append(float(log_csv[dom][alg]))
                gap_res.append(str(round(gap_lis[a]-gap_lis[b],2)))
            log_csv[f'{cols[a]}-{cols[b]}'] = gap_res

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
    print(trail_res_lis)



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
        # print(sdf)

        # CCS 
        ccs_dir = '/homes/55/jianhaoy/projects/EKI/results_Texture/ForTest'
        cols = ['Algorithm']
        for i in ['Texture_Bias','Shape_Bias']:
            cols.append(i)
        # cols.append('Average')
        log_csv = pd.DataFrame(columns=cols)
        log_csv['Algorithm'] = algo_list
        log_csv.set_index('Algorithm', inplace = True)
        for domain in ['original']:
            for algo in algo_list:
                # print(algo)
                log_path = f'{ccs_dir}/{algo}/{domain}.out'
                f = open(log_path,'r').readlines()[-1:]
                # print(f)
                for i in f:
                    # print(i)
                    test_acc = i.split(',')[-1].split(' ')[-1].strip('\\n')[:-2]
                    print(test_acc)
                    # print(toks)
                    test_domain = 'Texture_Bias'
                    test_acc = float(test_acc) * 100
                    tt = 100-test_acc
                    test_acc = str(round(test_acc, 2))
                    tt = str(round(tt, 2))
                    # print(test_domain,algo)
                    log_csv[test_domain][algo] = test_acc
                    log_csv['Shape_Bias'][algo] = tt

        # print(log_csv)
        ccs_csv = log_csv

        sdf['Texture_Bias'] = ccs_csv['Texture_Bias']
        sdf['Shape_Bias'] = ccs_csv['Shape_Bias']
        sdf.drop(columns=['Gap'],inplace=True)
        print(sdf)


    # # PRINT LINE
    rows = list(sdf.index)
    for alg in rows:
        line = f'{alg} '
        gap_lis = []
        for dom in list(sdf.columns):
            if dom in ['edge','silo','in-random']:
                continue
            cont = str(sdf[dom][alg])
            if str(cont) == 4:
                cont = str(cont)+'0'

            line += f'& ${cont}$ '
        line += ' \\\\'
        print(line)

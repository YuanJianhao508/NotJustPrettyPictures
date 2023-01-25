import pandas as pd
import numpy as np


source_list = ['art_painting','photo','sketch','cartoon']
# source_list = ['Art','Clipart','Product','Real_World']

algo_list = [f'{i}' for i in range(0,16)]

base_dir = '/homes/55/jianhaoy/projects/EKI/results/Multi_Samples_1'
trails = [f'/homes/55/jianhaoy/projects/EKI/results/Multi_Samples_{i}' for i in [1,2,3]]
single = False

# print(trail_lis)

if single:
    log_csv_lis = []
    for domain in ['photo']:
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

else:
    trail_res_lis = []
    log_csv_lis = []
    for trail in trails:
        for domain in ['photo']:
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
            print(log_csv_lis)
        trail_res_lis.append(log_csv_lis)

    cols = ['Algorithm']
    for i in source_list:
        if i != domain:
            cols.append(i)
    cols.append('Average')
    final_csv = pd.DataFrame(columns=cols)
    # final_csv['Algorithm'] = algo_list

    

    for log_csv in log_csv_lis:
        for ksample in range(16):
            acc_lis = [ksample]
            for domain in ['art_painting','sketch','cartoon','Average']:

                acc_lis.append(log_csv[domain][ksample])

            final_csv.loc[len(final_csv.index)] = acc_lis
    print(final_csv)
    final_csv.to_csv('/homes/55/jianhaoy/projects/EKI/results/ksample.csv')
                 
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
                mu = max(s)
                # mu,std = np.mean(s),np.std(s)
                # mu,std = str(round(mu, 2)),str(round(std, 2))
                # print(mu,std,col,row)
                # sdf[col][row] = f'{mu} \u00B1 {std}'
                sdf[col][row] = mu
        print(sdf)
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
            print(dom)
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
        for log_csv in sdf_lis:
            for dom in list(sdf.columns):
                # if dom == 'Average':
                #     continue
                line += f'& ${log_csv[dom][alg]}$ '
        line += ' \\\\'
        print(line)

    for d in ['art_painting','sketch','cartoon','Average']:
        print(list(sdf[d]))

    
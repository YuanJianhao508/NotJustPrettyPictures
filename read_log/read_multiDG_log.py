import pandas as pd
import numpy as np
import os


# source_list = ['art_painting','photo','sketch','cartoon']
source_list = ['Art','Clipart','Product','Real_World']

algos = ['Exclude','Include']
algo_list = [f"{i}/{j}" for i in algos for j in source_list]
# trails = [f'/homes/55/jianhaoy/projects/EKI/results/E30_PRET_CE{i}' for i in [1,2,3,4,5]]
base_dir = '/homes/55/jianhaoy/projects/EKI/results_Officehome/Multi_results/CIRL_handcrafted'

EI = True

# print(trail_lis)

# if EI:
#     log_csv_lis = []
#     cols = ['Algorithm']
#     for i in source_list:
#         cols.append(i)
#     cols.append('Average')
#     cols.append('Maximum')
#     log_csv = pd.DataFrame(columns=cols)
#     log_csv['Algorithm'] = algo_list
#     log_csv.set_index('Algorithm', inplace = True)
#     # print(log_csv)
#     for domain in source_list:
#         for algo in algo_list:
#             # print(algo)
#             log_path = f'{base_dir}/{algo}/{domain}.out'
#             if not os.path.exists(log_path):
#                 continue
#             f = open(log_path,'r').readlines()[-2:]
#             for i in f:
#                 # print(i)
#                 if '*' in i:
#                     continue
#                 toks = i.split(' ')
#                 # print(toks)

#                 test_domain = domain
#                 test_acc = toks[5].strip('\n')

#                 test_acc = float(test_acc) * 100
#                 test_acc = str(round(test_acc, 2))

#                 log_csv[test_domain][algo] = test_acc

#     for row in range(log_csv.shape[0]):
#         acc_sum = []
#         for col in source_list:
#             cont = log_csv.iloc[row][col]
#             if not pd.isna(cont):
#                 acc_sum.append(float(cont))
#         log_csv['Average'][algo_list[row]] = str(round(sum(acc_sum)/3,2))
#         log_csv['Maximum'][algo_list[row]] = str(round(max(acc_sum),2))

#     print(log_csv)

# else:
#     algo_list = ['CIRL_Plain','OURS_Multi']
#     base_dir = '/homes/55/jianhaoy/projects/EKI/results//Multi_results'
#     cols = ['Algorithm']
#     for i in source_list:
#         cols.append(i)
#     cols.append('Average')
#     log_csv = pd.DataFrame(columns=cols)
#     log_csv['Algorithm'] = algo_list
#     log_csv.set_index('Algorithm', inplace = True)
#     # print(log_csv)
#     for domain in source_list:
#         for algo in algo_list:
#             # print(algo)
#             log_path = f'{base_dir}/{algo}/{domain}.out'
#             f = open(log_path,'r').readlines()[-2:]
#             for i in f:
#                 # print(i)
#                 if '*' in i:
#                     continue
#                 toks = i.split(' ')
#                 # print(toks)

#                 test_domain = domain
#                 test_acc = toks[5].strip('\n')

#                 test_acc = float(test_acc) * 100
#                 test_acc = str(round(test_acc, 2))

#                 log_csv[test_domain][algo] = test_acc

#     # print(log_csv)
#     for row in range(log_csv.shape[0]):
#         acc_sum = 0
#         for col in source_list:
#             cont = log_csv.iloc[row][col]
#             acc_sum += float(cont)
#         log_csv['Average'][algo_list[row]] = str(round(acc_sum/4,2))

#     print(log_csv)



log_csv_lis = []
cols = ['Algorithm']
for i in source_list:
    cols.append(i)
cols.append('Average')
cols.append('Maximum')
log_csv = pd.DataFrame(columns=cols)
log_csv['Algorithm'] = algo_list
log_csv.set_index('Algorithm', inplace = True)
# print(log_csv)
for domain in source_list:
    for algo in algo_list:
        # print(algo)
        log_path = f'{base_dir}/{algo}/{domain}.out'
        print(log_path)
        if not os.path.exists(log_path):
            continue
        f = open(log_path,'r').readlines()[-2:]
        for i in f:
            # print(i)
            if '*' in i:
                continue
            toks = i.split(' ')
            # print(toks)

            test_domain = domain
            test_acc = toks[5].strip('\n')

            test_acc = float(test_acc) * 100
            test_acc = str(round(test_acc, 2))

            log_csv[test_domain][algo] = test_acc

for row in range(log_csv.shape[0]):
    acc_sum = []
    for col in source_list:
        cont = log_csv.iloc[row][col]
        if not pd.isna(cont):
            acc_sum.append(float(cont))
    log_csv['Average'][algo_list[row]] = str(round(sum(acc_sum)/3,2))
    # log_csv['Maximum'][algo_list[row]] = str(round(max(acc_sum),2))

print(log_csv)
EI_csv = log_csv


# algo_list = ['CIRL_Plain','OURS_Multi','OURS']
# base_dir = '/homes/55/jianhaoy/projects/EKI/results//Multi_results'

# cols = ['Algorithm']
# for i in source_list:
#     cols.append(i)
# cols.append('Average')
# log_csv = pd.DataFrame(columns=cols)
# log_csv['Algorithm'] = algo_list
# log_csv.set_index('Algorithm', inplace = True)
# # print(log_csv)
# for domain in source_list:
#     for algo in algo_list:
#         # print(algo)
#         log_path = f'{base_dir}/{algo}/{domain}.out'
#         f = open(log_path,'r').readlines()[-2:]
#         for i in f:
#             # print(i)
#             if '*' in i:
#                 continue
#             toks = i.split(' ')
#             # print(toks)

#             test_domain = domain
#             test_acc = toks[5].strip('\n')

#             test_acc = float(test_acc) * 100
#             test_acc = str(round(test_acc, 2))

#             log_csv[test_domain][algo] = test_acc

# # print(log_csv)
# for row in range(log_csv.shape[0]):
#     acc_sum = 0
#     for col in source_list:
#         cont = log_csv.iloc[row][col]
#         acc_sum += float(cont)
#     log_csv['Average'][algo_list[row]] = str(round(acc_sum/4,2))

# print(log_csv)
# t_csv = log_csv


algo_list = ['Exclude','Include']
# base_dir = '/homes/55/jianhaoy/projects/EKI/results//Multi_results'
cols = ['Algorithm']
for i in source_list:
    cols.append(i)
cols.append('Average')
log_csv = pd.DataFrame(columns=cols)
log_csv['Algorithm'] = algo_list
log_csv.set_index('Algorithm', inplace = True)
# print(log_csv)
algos = ['Exclude','Include']
algo_list = [f"{i}/{j}" for i in algos for j in source_list]
# idk = 'Average'
for idk in ["Average"]:
    print("*"*100)
    print(f"With {idk}")
    for row in algo_list:
        a = row.split('/')[0]
        b = row.split('/')[1]
        # print(a,b)
        log_csv[b][a] = EI_csv[idk][row]

    for row in range(log_csv.shape[0]):
        acc_sum = []
        for col in source_list:
            cont = log_csv.iloc[row][col]
            if not pd.isna(cont):
                acc_sum.append(float(cont))
        log_csv['Average'][algos[row]] = str(round(sum(acc_sum)/4,2))
    # print(log_csv)
    final_csv = log_csv
    # final_csv = log_csv.append(t_csv)
    print(final_csv)


    # PRINT LINE
    rows = list(final_csv.index)
    for alg in rows:
        line = f'{alg} '
        gap_lis = []
        for dom in list(final_csv.columns):
            # if dom == 'Average':
            #     continue
            # gap_lis.append(float(log_csv[dom][alg]))
            line += f'& ${final_csv[dom][alg]}$ '
        # line += f'& ${str(round(gap_lis[1]-gap_lis[0],2))}$ '
        line += ' \\\\'
        print(line)

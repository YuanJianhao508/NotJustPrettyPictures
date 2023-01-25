import pandas as pd

source_list = ['art_painting','photo','sketch','cartoon']
# source_list = ['Art','Clipart','Product','Real_World']

algo_list = ['OURS_20','OURS_40','OURS_60','OURS_80','OURS_100','OURS_120','OURS_140','OURS_160','OURS_180','OURS_200','OURS_220','OURS_240','OURS_260','OURS_280','OURS_300']
# algo_list = ['OURS_20','OURS_40','OURS_60']
base_dir = '/homes/55/jianhaoy/projects/EKI/results/multi_iter/dumb'
accuracy = False

# print(trail_lis)

if accuracy:
    di = {}
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
                # test_acc = str(round(test_acc, 2))
                test_acc = round(test_acc, 2)
                # print(algo,domain,'----->',test_domain,':',test_acc)
                # print(test_domain,algo)
                log_csv[test_domain][algo] = test_acc
        
        # print(log_csv)
        print(list(log_csv['Average'].to_numpy()))
        remain_domains = [i for i in source_list if i != domain]
        res_dict = {}
        res_dict.update({'Average':list(log_csv['Average'].to_numpy())})
        for i in remain_domains:
            res_dict.update({i:list(log_csv[i].to_numpy())})
        di.update({domain:res_dict})
    print(di)

else:
    base_dir = '/homes/55/jianhaoy/projects/EKI/results/multi_iter/measure_dumb/VGG-gram'
    algo_list = ['OURS_20','OURS_40','OURS_60','OURS_80','OURS_100','OURS_120','OURS_140','OURS_160','OURS_180','OURS_200','OURS_220','OURS_240','OURS_260','OURS_280','OURS_300']
    source_list = ['art_painting','photo','sketch','cartoon']
    di = {}
    for domain in source_list:
        print('-'*60)
        print(f'Source Domain:{domain}')
        cols = ['No.iteration']
        for i in source_list:
            if i != domain:
                cols.append(i)
        cols.append('Average')
        log_csv = pd.DataFrame(columns=cols)
        log_csv['No.iteration'] = algo_list
        log_csv.set_index('No.iteration', inplace = True)
        for log in algo_list:
            # print(algo_list)
            log_path = f'{base_dir}/{domain}/{log}.out'
            # print(log_path,domain,log)
            f = open(log_path,'r').readlines()[-8:]
            # print(log_path)
            for line in f:
                toks=line.split(' to ')
                if len(toks) < 2 and 'Average' not in line:
                    continue
                if 'Average' in line:
                    avr = float(toks[-1].split(' ')[-1].strip('\\n'))
                    log_csv['Average'][log] = avr
                else:
                    temp = toks[1].split(' ')
                    to_domain = temp[0]
                    score = float(temp[-1].strip('\\n'))
                    log_csv[to_domain][log] = score
        # print(log_csv)
        # print(list(log_csv['Average'].to_numpy()))
        remain_domains = [i for i in source_list if i != domain]
        res_dict = {}
        res_dict.update({'Average':list(log_csv['Average'].to_numpy())})
        for i in remain_domains:
            res_dict.update({i:list(log_csv[i].to_numpy())})
        di.update({domain:res_dict})
    print(di)
        


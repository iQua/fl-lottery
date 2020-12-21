# -*- coding: utf-8 -*-
import json
import csv
import os
import logging
import matplotlib.pyplot as plt
import matplotlib




root_path = '07_29_12:36:17-lottery'


class ClientDataReport(object):
    '''
    params: 
    client_id
    accuracy 
    
    '''
    def __init__(self, round_id, client_id):
        self.client_id = client_id
        self.round_id = round_id
        self.accuracy = []
        self.tot_params = 0
        self.tot_size = 0.0
        #from level 1-12
        self.ep0_unpruned_list = []
        self.final_unpruned_list = []
        
    def set_accuracy(self, acc_list):
        self.accuracy = acc_list
    
    def set_tot_size(self, params_size):
        self.tot_size = params_size
    
    def set_tot_params(self, tot_params):
        self.tot_params = tot_params

def plot_acc_pic(cdreport_dict):
    path = root_path +f'/pics'
    if not os.path.exists(path):
            os.mkdir(path)

    line_style = ['-','--',':']

    for round_id, cdreport_list in cdreport_dict.items():
        i = 0
        for cdreport in cdreport_list:
            i += 1
            acc = cdreport.accuracy
            levels = list(range(len(acc)))
            plt.plot(levels, acc, line_style[i%3], label=cdreport.client_id)
        plt.xlabel('Levels')
        plt.ylabel('Accuracy')
        plt.ylim(90,100)
        plt.legend(loc='upper left',fontsize='x-small',prop={'size': 6})

        
        pic_path = root_path + f'/pics/round{round_id}_acc.png'
        if os.path.exists(pic_path):
            os.remove(pic_path)
        plt.savefig(pic_path)
        plt.close()


def get_global_report(round_id, round_path):
    global_path = os.path.join(round_path, 'global')  
    
    greport = ClientDataReport(round_id, 'global')
    #get accuracy from accuracy.json
    with open(os.path.join(round_path, 'accuracy.json'), 'r') as f:
        gdict = json.load(f)
        glist = [round(float(acc)*100,2) for _, acc in gdict.items()]
        greport.set_accuracy(glist)
    #deal with levels and the summary
    logging.info('Global Model')
    level_list = get_child_dir(global_path, True)
    
    for level in level_list:
        level = os.path.join(global_path, level)
        summary = level + f'/modelsummary'
        sparsity = level + f'/sparsity_report.json'
        
        greport.tot_size = get_tot_size(summary)
        if os.path.exists(sparsity):
            (tot_params, num) = get_sparsity(sparsity)
        else:
            num = float("NAN")
        greport.tot_params = tot_params
        greport.ep0_unpruned_list.append(num)
        greport.final_unpruned_list.append(num)
    output_in_log(greport)

    return greport

def output_in_log(cdr: ClientDataReport):
    for i in range(len(cdr.ep0_unpruned_list)):
        num = int(cdr.ep0_unpruned_list[i])
        fnum = cdr.final_unpruned_list[i]
        rate = round(num / cdr.tot_params, 4)
        frate = round(fnum / cdr.tot_params, 4)
        #size = round(rate * cdr.tot_size, 4)
        logging.info(f'Level {i}: unpruned at ep0: {num} rate: {rate}, unpruned at last ep: {fnum} rate:{frate}')

    
def get_child_dir(path, choosedir):
    child_dir = []
    for name in os.listdir(path):
        if choosedir:
            if os.path.isdir(os.path.join(path, name)):
                child_dir.append(name)
        else:
            child_dir.append(name)
    child_dir = sorted(child_dir)
    return child_dir

def get_tot_size(path):
    f = open(path, 'r')
    for line in f:
        if 'Estimated Total Size' in line:
            tot_size = float(line[27:31])
            return tot_size
    f.close()
    return 0.0

def get_sparsity(path):
    with open(path, 'r')as f:
        sparsity_dict = json.load(f)
    tot_params = sparsity_dict['total']
    unpruned = sparsity_dict['unpruned']
    
    return (tot_params, unpruned)



log_path = os.path.join(root_path,'log')
if os.path.exists(log_path):
    os.remove(log_path)
os.mknod(log_path)

logging.basicConfig(filename=log_path, format='%(message)s', level=logging.INFO)

round_id_list = get_child_dir(root_path, True)
#print(round_id_list)
cdreport_dict = {}

#save data to reports
for round_id in round_id_list:
    if round_id == "pics":
        continue
    logging.info(f'\nRound {round_id}.\n')

    round_path = os.path.join(root_path, round_id)
    client_list = get_child_dir(round_path, True)
    
    cdreport_list = []

    for client_id in client_list:
        
        clt_acc = []
        clt_size = []
        cdreport = ClientDataReport(round_id, client_id)

        if client_id == 'global':
            continue
        
        logging.info(f'Client {client_id}')
        client_path = os.path.join(round_path, client_id)
        
        for name in os.listdir(client_path):
            client_path = os.path.join(client_path, name, 'replicate_1')
        
        #level1-12
        level_list = get_child_dir(client_path, True)
        
        for i in range(len(level_list)):
            #set all paths
            level = level_list[i]
            folder_path = os.path.join(client_path, level, 'main')
            src = os.path.join(folder_path, 'logger')
            log = os.path.join(folder_path, 'logger.csv')
          
            for name in os.listdir(folder_path):
                if 'summary' in name:
                    summary = os.path.join(folder_path, name)
            sparsity = os.path.join(folder_path, 'sparsity_report.json')
            fs = os.path.join(folder_path, 'sparsity_report_after_training.json')
            #accuracy 
            if os.path.exists(src):
                os.rename(src, log)

            with open(log, 'r+') as cf:
                csv_reader = csv.reader(cf, delimiter=' ')
                accuracy_list = []
                step = 1
                for row in csv_reader:
                    if(step%4==2):
                        data = row[0].split(",")
                        accuracy_list.append(data[2])
                    step = step+1
            accuracy = [round(float(item)*100,2) for item in accuracy_list]
            clt_acc.append(max(accuracy))
            
            #total params and sparsity
            cdreport.tot_size = get_tot_size(summary)
            tot_params, num =get_sparsity(sparsity)
            _, fnum = get_sparsity(fs)
            cdreport.tot_params = tot_params
            cdreport.ep0_unpruned_list.append(num)
            cdreport.final_unpruned_list.append(fnum)

        output_in_log(cdreport)
        cdreport.set_accuracy(clt_acc)
        cdreport_list.append(cdreport)

    globalreport = get_global_report(round_id, round_path)
    logging.info(f'total model size: {globalreport.tot_size} MB, total parmas: {globalreport.tot_params}.')
    cdreport_list.append(globalreport)
    cdreport_dict[round_id] = cdreport_list
    


#plot and output result
plot_acc_pic(cdreport_dict)



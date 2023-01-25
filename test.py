import argparse
import ast
from collections import deque
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from models.model_factory import *
from data import *
from utils.Logger import Logger
from utils.tools import *
from utils.AdaECE import AdaECE
from utils.ClassificationRejection import calculate_rejection


import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--input_dir", default=None, help="The directory of dataset lists")
    parser.add_argument("--output_dir", default=None, help="The directory to save logs and models")
    parser.add_argument("--config", default=None, help="Experiment configs")
    parser.add_argument("--tf_logger", type=ast.literal_eval, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--ckpt", default=None, help="The directory to models")
    parser.add_argument("--algo",default=None,help="Which Algo")
    args = parser.parse_args()
    config_file = "config." + args.config.replace("/", ".")
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config

class Evaluator:
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        self.global_step = 0
        print(self.args.ckpt)

        if self.args.algo == 'SN':
            self.config["networks"]["encoder"]["name"] = "resnet18_sn"
        elif self.args.algo == 'SRN':
            self.config["networks"]["encoder"]["name"] = "resnet18_srn"

        # networks
        self.encoder = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.classifier = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)

        # dataloaders
        # self.val_loader = get_single_val_dataloader(args=self.args, config=self.config)
        self.test_loader = get_single_test_loader(args=self.args, config=self.config)
        self.adap_test_loader = get_adaptation_test_loader(args=self.args, config=self.config)
        
        self.separate_test_loader = get_separate_test_loader(args=self.args, config=self.config)
        self.separate_adap_test_loader = get_separate_adap_test_loader(args=self.args, config=self.config)
        self.eval_loader = {'test': self.test_loader,'adaptation':self.adap_test_loader}
        self.eval_loader.update(self.separate_test_loader)
        self.eval_loader.update(self.separate_adap_test_loader)

        self.logits = torch.tensor([]).to(device)
        self.labels = torch.tensor([]).to(device)
        self.confidence = torch.tensor([]).to(device)
        self.preds = torch.tensor([]).to(device)
        

        self.softmax = nn.Softmax(dim=1)

    def do_eval(self, loader):
        correct = 0
        for it, (batch, domain) in enumerate(loader):
            data, labels, domains = batch[0].to(self.device), batch[1].to(self.device), domain.to(self.device)
            if self.args.target in pacs_dataset:
                labels -= 1
            features = self.encoder(data)
            scores = self.classifier(features)
            scores = self.softmax(scores)
            # print(scores.size())
            confidence = torch.max(scores,1)[0]
            _, preds = scores.max(dim=1)
            # print(pred)
            # print(confidence)
            # print(confidence.size())
            self.logits = torch.cat((self.logits,scores),0)
            self.labels = torch.cat((self.labels,labels),0)
            self.confidence = torch.cat((self.confidence,confidence),0)
            self.preds = torch.cat((self.preds,preds),0)
            # print(self.logits,self.labels,self.confidence)
            # print(self.logits.size(),self.labels.size(),self.confidence.size())
            correct += calculate_correct(scores, labels)

        return correct

    def do_testing(self):
        self.logger = Logger(self.args, self.config, update_frequency=30)

        if self.args.ckpt is not None:
            state_dict = torch.load(self.args.ckpt, map_location=lambda storage, loc: storage)
            encoder_state = state_dict["encoder_state_dict"]
            classifier_state = state_dict["classifier_state_dict"]
            self.encoder.load_state_dict(encoder_state)
            self.classifier.load_state_dict(classifier_state)
        
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            for phase, loader in self.eval_loader.items():
                total = len(loader.dataset)
                class_correct = self.do_eval(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {'class': class_acc})

    def calculate_AdaECE(self):
        out = AdaECE().to(self.device)(self.logits, self.labels)
        print(f"AdaECE:{out}")
        return float(out)

    def calculate_rejection(self):
        # print(self.preds)
        # print(self.logits.size(),self.labels.size(),self.confidence.size())
        rejection_ratio = calculate_rejection(self.preds,self.labels,self.confidence)
        print(f'Rejection:{rejection_ratio}')
        return rejection_ratio

    def calculate_texture_bias(self):
        index_to_name = {0: 'airplane', 1: 'bear', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'dog', 7: 'cat', 8: 'car', 9: 'clock', 10: 'chair', 11: 'elephant', 12: 'keyboard', 13: 'knife', 14: 'oven', 15: 'truck'}
        name_to_index = {val:key for key,val in index_to_name.items()}

        _, pred = self.logits.max(dim=1)
        class_label = self.labels
        domain_label = []
        file_name = f'random_test.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        for line in lines:
            toks = line.split(' ')[0].split('/')[-1].split('-')[-1].strip('.png').rstrip('0123456789')
            domain_index = int(name_to_index[toks])
            domain_label.append(domain_index)
        domain_label = torch.from_numpy(np.array(domain_label)).to(self.device)
        # print(class_label,domain_label)
        shape_correct = torch.sum(pred.eq(class_label)).item()

        # shape_acc = shape_correct/len(class_label)
        texture_correct = torch.sum(pred.eq(domain_label)).item()
        # texture_acc = texture_correct/len(class_label)
        texture_bias = texture_correct / (texture_correct + shape_correct)
        print(texture_bias)
        return texture_bias

        # print(acc)
        # acc = sum(correct)/len(correct)

        


def main():
    args, config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_ckpt_lis = args.ckpt.split('-.-')
    res = {'AdaECE':[],'Rejection':[],'TextureBias':[1]}
    for i in range(1):
        args.ckpt = save_ckpt_lis[i]
        evaluator = Evaluator(args, config, device)
        evaluator.do_testing()
        ada = evaluator.calculate_AdaECE()
        rej = evaluator.calculate_rejection()
        # tb = evaluator.calculate_texture_bias()
        res['AdaECE'].append(ada)
        # res['TextureBias'].append(tb)
        res['Rejection'].append(rej)
        # print(res)
    for key,val in res.items():
        res[key] = sum(val)/len(val)
    print(res)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
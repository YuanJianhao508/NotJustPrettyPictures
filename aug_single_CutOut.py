import argparse
import ast
from collections import deque
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from models.model_factory import *
from optimizer.optimizer_helper import get_optim_and_scheduler
from data import *
from utils.Logger import Logger
from utils.tools import *
from models.classifier import Masker

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
    parser.add_argument("--link_dict",default=None,help="link file for intervention image")
    parser.add_argument("--test_domain",default=None,help="test domain")
    parser.add_argument("--tf_logger", type=ast.literal_eval, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--ifcons",type=str,help='Whether to use Contrastive loss')
    args = parser.parse_args()
    config_file = "config." + args.config.replace("/", ".")
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config


class Trainer:
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        self.global_step = 0

        # networks
        self.encoder = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.classifier = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)
       
        # optimizers
        self.encoder_optim, self.encoder_sched = \
            get_optim_and_scheduler(self.encoder, self.config["optimizer"]["encoder_optimizer"])
        self.classifier_optim, self.classifier_sched = \
            get_optim_and_scheduler(self.classifier, self.config["optimizer"]["classifier_optimizer"])

        # dataloaders
        self.train_loader = get_single_cutout_train_dataloader(args=self.args, config=self.config)
        # self.train_loader = get_fourier_train_dataloader(args=self.args, config=self.config)
        self.val_loader = get_single_val_dataloader(args=self.args, config=self.config)
        self.test_loader = get_single_test_loader(args=self.args, config=self.config)
        self.eval_loader = {'val': self.val_loader, 'test': self.test_loader}
        self.separate_test_loader = get_separate_test_loader(args=self.args, config=self.config)

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()

        # turn on train mode
        self.encoder.train()
        self.classifier.train()

        for it, (batch, label, domain) in enumerate(self.train_loader):

            # preprocessing
            batch = torch.cat(batch, dim=0).to(self.device)
            labels = torch.cat(label, dim=0).to(self.device)
            if self.args.target in pacs_dataset:
                labels -= 1
            # zero grad
            self.encoder_optim.zero_grad()
            self.classifier_optim.zero_grad()

            # forward
            loss_dict = {}
            correct_dict = {}
            num_samples_dict = {}
            total_loss = 0.0

            ## --------------------------step 1 : update G and C -----------------------------------
            features = self.encoder(batch)
            scores = self.classifier(features)

            loss_cls = criterion(scores, labels)
            loss_dict["cls"] = loss_cls.item()
            correct_dict["cls"] = calculate_correct(scores, labels)
            num_samples_dict["cls"] = int(scores.size(0))


            if self.args.ifcons == "ceclip":
                assert batch.size(0) % 2 == 0
                split_idx = int(batch.size(0) / 2)
                features_ori, features_aug = torch.split(features, split_idx)
                assert features_ori.size(0) == features_aug.size(0)
                # factorization loss for features between ori and aug
                loss_fac = clip_loss_v2(features_ori,features_aug,self.device)
                loss_dict["fac"] = loss_fac.item()

                # calculate total loss
                total_loss = loss_cls + loss_fac
                
            elif self.args.ifcons == 'ce':
                total_loss = loss_cls

            loss_dict["total"] = total_loss.item()
            # backward
            total_loss.backward()

            # update
            self.encoder_optim.step()
            self.classifier_optim.step()

            self.global_step += 1

            # record
            self.logger.log(
                it=it,
                iters=len(self.train_loader),
                losses=loss_dict,
                samples_right=correct_dict,
                total_samples=num_samples_dict
            )

        # turn on eval mode
        self.encoder.eval()
        self.classifier.eval()

        # evaluation
        if self.current_epoch >= 40:
            with torch.no_grad():
                for phase, loader in self.eval_loader.items():
                    if phase != "test":
                        total = len(loader.dataset)
                        class_correct = self.do_eval(loader)
                        class_acc = float(class_correct) / total
                        self.logger.log_test(phase, {'class': class_acc})
                        self.results[phase][self.current_epoch] = class_acc
                    else:
                        self.logger.log_test(phase, {'class': 0.1})
                        self.results[phase][self.current_epoch] = 0.1
                # save from best val
                if self.results['val'][self.current_epoch] >= self.best_val_acc:
                    self.best_val_acc = self.results['val'][self.current_epoch]
                    self.best_val_epoch = self.current_epoch + 1
                    self.logger.save_best_model(self.encoder, self.classifier, self.best_val_acc)

                for test_domain, loader in self.separate_test_loader.items():
                    total = len(loader.dataset)
                    class_correct = self.do_eval(loader)
                    class_acc = float(class_correct) / total
                    self.logger.log_test(test_domain, {'test class': class_acc})
                    self.results[test_domain][self.current_epoch] = class_acc

    def do_eval(self, loader):
        correct = 0
        for it, (batch, domain) in enumerate(loader):
            data, labels, domains = batch[0].to(self.device), batch[1].to(self.device), domain.to(self.device)
            if self.args.target in pacs_dataset:
                labels -= 1
            features = self.encoder(data)
            #Add
            scores = self.classifier(features)
            correct += calculate_correct(scores, labels)
        return correct


    def do_training(self):
        self.logger = Logger(self.args, self.config, update_frequency=30)
        self.logger.save_config()

        self.epochs = self.config["epoch"]
        self.results = {"val": torch.zeros(self.epochs), "test": torch.zeros(self.epochs)}
        self.results.update({key:torch.zeros(self.epochs) for key in self.args.source})

        self.best_val_acc = 0
        self.best_val_epoch = 0

        for self.current_epoch in range(self.epochs):

            # step schedulers
            self.encoder_sched.step()
            self.classifier_sched.step()

            self.logger.new_epoch([group["lr"] for group in self.encoder_optim.param_groups])
            self._do_epoch()
            self.logger.finish_epoch()

        # save from best val
        val_res = self.results['val']
        test_res = self.results['test']
        self.logger.save_best_acc(val_res, test_res, self.best_val_acc, self.best_val_epoch - 1)
        for test_domain in self.args.source:
            single_res = self.results[test_domain]
            self.logger.save_single_best_acc(val_res, single_res, self.best_val_acc, self.best_val_epoch - 1, test_domain)

        return self.logger


def main():
    args, config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, config, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
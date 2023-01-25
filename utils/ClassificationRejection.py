from torchmetrics import Metric
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import auc
#preds = tensor containing the predicted labels for each element in the batch
#labels = tensor containing all the TRUE labels for each element in the batch
#confidence = tensor containing the maximum probability score of the softmax for each element in the batch
def calculate_rejection(preds, labels, confidence):
        #based on https://github.com/KaosEngineer/PriorNetworks/blob/master/prior_networks/assessment/rejection.py
        # uncertainty_results = self.collect_uncertainties(outputs, prefix)
        #compute area between base_error(1-x) and the rejection curve
        #compute area between base_error(1-x) and the oracle curve
        #take the ratio
        #preds represents the predicted label, compute it in whichever way you need to
        #the rejection plots needs to reject to the left the most uncertain/less confident samples
        #if uncertainty metric, high means reject, sort in descending uncertainty; if confidence metric, low means reject, sort in ascending confidence
        descending = True
        # plots to add:
        # - histogram of uncertainties for IND/OOD .
        # - histogram of uncertainties for correctly classified/wrongly classified
        # - plot aupr/auroc plots
        # - plot reliability plots
        # - plot
        # descending dovrebbe riguardare anche gli ood score di dirichlet
        sorted_idx = torch.argsort(confidence, descending = descending)
        #reverse cumulative errors function (rev = from all to first, instead from first error to all)
        rev_cum_errors = []
        #fraction of data rejected, to compute a certain value of rev_cum_errors
        fraction_data = []
        num_samples = preds.shape[0]
        for i in range(1,num_samples):
            rev_cum_errors.append(torch.tensor([torch.sum(labels[sorted_idx[:i]] != preds[sorted_idx[:i]]).float()*100.0/float(num_samples)]))
            fraction_data.append(torch.tensor([float(i + 1) / float(num_samples) * 100.0]))
        rev_cum_errors, fraction_data = torch.cat(rev_cum_errors, dim=0).cpu().numpy(), torch.cat(fraction_data, dim=0).cpu().numpy()
        # import pdb; pdb.set_trace()
        base_error = rev_cum_errors[-1] #min error possible
        n_items = rev_cum_errors.shape[0]
        #area under the rejection curve (used later to compute area between random and rejection curve)
        auc_uns = 1.0 - auc(fraction_data / 100.0, rev_cum_errors[::-1] / 100.0)
        #random rejection baseline, it's 1 - x line "scaled" and "shifted" to pass through base error and go to 100% rejection
        random_rejection = np.asarray(
            [base_error * (1.0 - float(i) / float(n_items)) for i in range(n_items)],
            dtype=np.float32)
        #area under random rejection, should be 0.5
        auc_rnd = 1.0 - auc(fraction_data / 100.0, random_rejection / 100.0)
        #oracle curve, the oracle is assumed to commit the base error
        #making the oracle curve commit the base error allows to remove the impact of the base error when computing
        #the ratio of areas
        #line passing through base error at perc_rej = 0, and crossing
        #the line goes from x=0 to x=base_error/100*n_items <- this is when the line intersects the x axis
        #which means the oracle ONLY REJECTS THE SAMPLES THAT ARE MISCASSIFIED
        #afterwards the function is set to zero
        orc_rejection = np.asarray(
            [base_error * (1.0 - float(i) / float(base_error / 100.0 * n_items)) for i in
                range(int(base_error / 100.0 * n_items))], dtype=np.float32)
        orc = np.zeros_like(rev_cum_errors)
        orc[0:orc_rejection.shape[0]] = orc_rejection
        auc_orc = 1.0 - auc(fraction_data / 100.0, orc / 100.0)
        # random_rejection = np.squeeze(random_rejection)
        # orc = np.squeeze(orc)
        # rev_cum_errors = np.squeeze(rev_cum_errors)
        # import matplotlib.pyplot as plt
        # plt.plot(fraction_data, orc, lw=2)
        # plt.fill_between(fraction_data, orc, random_rejection, alpha=0.5)
        # plt.plot(fraction_data, rev_cum_errors[::-1], lw=2)
        # plt.fill_between(fraction_data, rev_cum_errors[::-1], random_rejection, alpha=0.0)
        # plt.plot(fraction_data, random_rejection, 'k--', lw=2)
        # plt.legend(['Oracle', 'Uncertainty', 'Random'])
        # plt.xlabel('Percentage of predictions rejected to oracle')
        # plt.ylabel('Classification Error (%)')
        # plt.savefig(f'/Rejection-Curve-oracle{k}.png', bbox_inches='tight', dpi=300)
        # # plt.show()
        # plt.cla()
        # plt.plot(fraction_data, orc, lw=2)
        # plt.fill_between(fraction_data, orc, random_rejection, alpha=0.0)
        # plt.plot(fraction_data, rev_cum_errors[::-1], lw=2)
        # plt.fill_between(fraction_data, rev_cum_errors[::-1], random_rejection, alpha=0.5)
        # plt.plot(fraction_data, random_rejection, 'k--', lw=2)
        # plt.legend(['Oracle', 'Uncertainty', 'Random'])
        # plt.xlabel('Percentage of predictions rejected to oracle')
        # plt.ylabel('Classification Error (%)')
        # plt.savefig(f'/Rejection-Curve-uncertainty{k}.png', bbox_inches='tight', dpi=300)
        # # plt.show()
        # plt.cla()
        #reported from -100 to 100
        rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0
        # print("PRR confidence", rejection_ratio)
        return rejection_ratio
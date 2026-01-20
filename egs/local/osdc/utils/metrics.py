import numpy as np
import torch
from sklearn.metrics import average_precision_score

import torch
import numpy as np

class BinaryMeter(object):

    def __init__(self):
        
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.acc = 0
        self.der = 0
        self.current_output = None
        self.current_target = None

    def update(self, output, target):
        # print(f'output:{output.shape}')
        # print(f'output:{output}')
        # print(f'target:{target}')
        # print(f'target:{target}')

        pred = (output >= 0.5).float()
        truth = (target >= 0.5).float()

        # print(f'pred:{pred.shape}')
        # print(f'truth:{truth.shape}')

        self.tp += pred.mul(truth).sum().float()
        self.tn += (1 - pred).mul(1 - truth).sum().float()
        self.fp += pred.mul(1 - truth).sum().float()
        self.fn += (1 - pred).mul(truth).sum().float()
        # print(f'self.tp:{self.tp}')
        # print(f'self.tn:{self.tn}')
        # print(f'self.fp:{self.fp}')
        # print(f'self.fn:{self.fn}')

        self.current_output = output
        self.current_target = target

    def get_fa(self):
        return self.fp / (self.get_positive_examples() + np.finfo(np.float32).eps)

    def get_FA(self):
        return self.fp / (self.get_negative_examples() + np.finfo(np.float32).eps)
# ---------------------------------------------------------------------------
    def get_MR(self):
        return self.fn / (self.get_positive_examples() + np.finfo(np.float32).eps)
    
    def get_FAR(self):
        return self.fp / (self.get_negative_examples() + np.finfo(np.float32).eps)

    def get_HTER(self):
        print(f'VAD:')
        print(f'真正例:{self.tp}')
        print(f'真反例:{self.tn}')
        print(f'假正例:{self.fp}')
        print(f'假反例:{self.fn}')
        
        # print(f'VAD 总样本数：{self.tp + self.tn + self.fp + self.fn}')
        return (self.get_MR() + self.get_FAR())/2.0
# ---------------------------------------------------------------------------
    def get_tp(self):
        return self.tp / (self.get_positive_examples() + np.finfo(np.float32).eps)

    def get_tn(self):
        return self.tn / (self.get_positive_examples() + np.finfo(np.float32).eps)

    def get_miss(self):
        return self.fn / (self.get_positive_examples() + np.finfo(np.float32).eps)

    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def get_precision(self):
        return self.tp / (self.tp + self.fp + np.finfo(np.float32).eps)

    def get_recall(self):
        return self.tp / (self.tp + self.fn + np.finfo(np.float32).eps)

    def get_f1(self):
        print(f'OSD:')
        print(f'真正例:{self.tp}')
        print(f'真反例:{self.tn}')
        print(f'假正例:{self.fp}')
        print(f'假反例:{self.fn}')
        
        print(f'总样本数：{self.tp + self.tn + self.fp + self.fn}')

        return (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn)

    def get_der(self):
        return (self.fp + self.fn) / (self.fn + self.tp + np.finfo(np.float32).eps)

    def get_mattcorr(self):
        return (self.tp * self.tn - self.fp * self.fn) / \
               (np.finfo(np.float32).eps + (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (
                       self.tn + self.fn)).sqrt()

    def get_tot_examples(self):
        return self.tp + self.tn + self.fp + self.fn

    def get_positive_examples(self):
        return self.fn + self.tp
    
    def get_negative_examples(self):
        return self.fp + self.tn

   



class MultiMeter(object):
    """Macro  metrics"""

    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.reset()


    def reset(self):
        self.tp = [0]*self.n_classes
        self.tn = [0]*self.n_classes
        self.fp = [0]*self.n_classes
        self.fn = [0]*self.n_classes
        self.acc = 0
        self.der = 0
      

    def update(self, output, target):
  
        for i in range(self.n_classes): # iterate over all classes
            pred = (output.float() == i).float()
            truth = (target.float() == i).float()
            self.tp[i] += pred.mul(truth).sum().float()
            self.tn[i] += (1. - pred).mul(1. - truth).sum().float()
            self.fp[i] += pred.mul(1. - truth).sum().float()
            self.fn[i] += (1. - pred).mul(truth).sum().float()
      


    def get_tp(self):
        return torch.sum(torch.stack(self.tp), 0)


    def get_tn(self):
        return torch.sum(torch.stack(self.tn), 0)


    def get_fp(self):
        return torch.sum(torch.stack(self.fp), 0)

    def get_fn(self):
        return torch.sum(torch.stack(self.fn), 0)


    def get_fa(self):

        fa = []
        for i in range(self.n_classes):
            fa.append(self.fp[i] / (self.get_positive_examples() + np.finfo(np.float64).eps))

        return torch.mean(torch.stack(fa), 0)

    def get_miss(self):


        miss = []

        for i in range(self.n_classes):

            miss.append(self.fn[i] / (self.get_positive_examples() + np.finfo(np.float64).eps))

        return torch.mean(torch.stack(miss), 0)

    def get_accuracy(self):

        acc = []
        for i in range(self.n_classes):
            acc.append((self.tp[i] + self.tn[i]) / (self.tp[i] + self.tn[i] + self.fp[i] + self.fn[i]))

        return torch.mean(torch.stack(acc), 0)

    def get_precision(self):

        prec = []
        for i in range(self.n_classes):
            prec.append(self.tp[i] / (self.tp[i]+ self.fp[i] + np.finfo(np.float64).eps))

        return torch.mean(torch.stack(prec), 0)

    def get_recall(self):

        recall = []
        for i in range(self.n_classes):
            recall.append(self.tp[i] / (self.tp[i]+ self.fn[i] + np.finfo(np.float64).eps))

        return torch.mean(torch.stack(recall), 0)


    def get_f1(self):

        f1 = []
        for i in range(self.n_classes):
            f1.append((2.0 * self.tp[i]) / (2.0 * self.tp[i] + self.fp[i] + self.fn[i]))

        return torch.mean(torch.stack(f1), 0)

    def get_der(self):

        der = []

        for i in range(self.n_classes):
            der.append( (self.fp[i] + self.fn[i]) / (self.fn[i] + self.tp[i] + np.finfo(np.float64).eps))

        return torch.mean(torch.stack(der), 0)

    def get_matt(self):

        matt = []

        for i in range(self.n_classes):
            matt.append((self.tp[i] * self.tn[i] - self.fp[i] * self.fn[i]) / \
               (np.finfo(np.float).eps + (self.tp[i] + self.fp[i]) * (self.tp[i] + self.fn[i]) * (self.tn[i] + self.fp[i]) * (
                           self.tn[i] + self.fn[i])).sqrt()  )

        return torch.mean(torch.stack(matt), 0)

    def get_tot_examples(self):

        tot = []
        for i in range(self.n_classes):
            tot.append( self.tp[i] + self.tn[i] + self.fp[i] + self.fn[i])

        return torch.sum(torch.stack(tot), 0)

    def get_positive_examples(self):
        tot = []
        for i in range(self.n_classes):
            tot.append(self.tp[i] + self.fn[i])

        return torch.sum(torch.stack(tot), 0)

    def get_positive_examples_class(self, i):
        return self.tp[i] + self.fn[i]

    # def get_class_aps(self,preds,label):  
    #     #  Convert list of outputs and targets to numpy arrays
    #     preds = preds.detach().cpu().numpy()
    #     labels = label.detach().cpu().numpy()

    #     # Reshape preds and labels to match the required format
    #     preds = preds.transpose(0, 2, 1).reshape(-1, self.n_classes)  # [batch_size * seq_len, n_classes]
    #     labels = labels.reshape(-1)  # [batch_size * seq_len]

    #     # Initialize list to store AP scores for each class
    #     ap_scores = []
 
    #     # Compute AP for each class
    #     for i in range(self.n_classes):
    #         # Get binary true labels and predictions for class i
    #         binary_targets = (labels == i).astype(int)
    #         binary_outputs = preds[:, i]

    #         # Compute average precision score for this class
    #         if np.sum(binary_targets) == 0:  # No positive samples for this class
    #             ap_scores.append(0.0)
    #         else:
    #             # ap = average_precision_score(binary_targets, binary_outputs,average='micro')
    #             ap = average_precision_score(binary_targets, binary_outputs)
    #             ap_scores.append(ap)

    #     # Convert the list of AP scores to a tensor
    #     return torch.tensor(ap_scores)
   






# import numpy as np
# import torch

# class BinaryMeter(object):

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.tp = 0
#         self.tn = 0
#         self.fp = 0
#         self.fn = 0
#         self.acc = 0
#         self.der = 0

#     def update(self, output, target):

#         pred = (output >= 0.5).float()
#         truth = (target >= 0.5).float()

#         self.tp += pred.mul(truth).sum().float()
#         self.tn += (1 - pred).mul(1 - truth).sum().float()
#         self.fp += pred.mul(1 - truth).sum().float()
#         self.fn += (1 - pred).mul(truth).sum().float()

#     def get_fa(self):
#         return self.fp / (self.get_positive_examples() + np.finfo(np.float64).eps)

#     def get_tp(self):
#         return self.tp / (self.get_positive_examples() + np.finfo(np.float64).eps)

#     def get_tn(self):
#         return self.tn / (self.get_positive_examples() + np.finfo(np.float64).eps)

#     def get_miss(self):
#         return self.fn / (self.get_positive_examples() + np.finfo(np.float64).eps)

#     def get_accuracy(self):
#         return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

#     def get_precision(self):
#         return self.tp / (self.tp + self.fp + np.finfo(np.float64).eps)

#     def get_recall(self):
#         return self.tp / (self.tp + self.fn + np.finfo(np.float64).eps)

#     def get_f1(self):
#         return (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn)

#     def get_der(self):
#         return (self.fp + self.fn) / (self.fn + self.tp + np.finfo(np.float64).eps)

#     def get_mattcorr(self):
#         return (self.tp * self.tn - self.fp * self.fn) / \
#                (np.finfo(np.float).eps + (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (
#                        self.tn + self.fn)).sqrt()
#         # matthew correlation coeff balanced measure for even unbalanced data

#     def get_tot_examples(self):
#         return self.tp + self.tn + self.fp + self.fn

#     def get_positive_examples(self):
#         return self.fn + self.tp


# class MultiMeter(object):
#     """Macro  metrics"""

#     def __init__(self, n_classes=5):
#         self.n_classes = n_classes
#         self.reset()


#     def reset(self):
#         self.tp = [0]*self.n_classes
#         self.tn = [0]*self.n_classes
#         self.fp = [0]*self.n_classes
#         self.fn = [0]*self.n_classes
#         self.acc = 0
#         self.der = 0

#     def update(self, output, target, ):

#         for i in range(self.n_classes): # iterate over all classes
#             pred = (output.float() == i).float()
#             truth = (target.float() == i).float()
#             self.tp[i] += pred.mul(truth).sum().float()
#             self.tn[i] += (1. - pred).mul(1. - truth).sum().float()
#             self.fp[i] += pred.mul(1. - truth).sum().float()
#             self.fn[i] += (1. - pred).mul(truth).sum().float()


#     def get_tp(self):
#         return torch.sum(torch.stack(self.tp), 0)


#     def get_tn(self):
#         return torch.sum(torch.stack(self.tn), 0)


#     def get_fp(self):
#         return torch.sum(torch.stack(self.fp), 0)

#     def get_fn(self):
#         return torch.sum(torch.stack(self.fn), 0)


#     def get_fa(self):

#         fa = []
#         for i in range(self.n_classes):
#             fa.append(self.fp[i] / (self.get_positive_examples() + np.finfo(np.float).eps))

#         return torch.mean(torch.stack(fa), 0)

#     def get_miss(self):


#         miss = []

#         for i in range(self.n_classes):

#             miss.append(self.fn[i] / (self.get_positive_examples() + np.finfo(np.float).eps))

#         return torch.mean(torch.stack(miss), 0)

#     def get_accuracy(self):

#         acc = []
#         for i in range(self.n_classes):
#             acc.append((self.tp[i] + self.tn[i]) / (self.tp[i] + self.tn[i] + self.fp[i] + self.fn[i]))

#         return torch.mean(torch.stack(acc), 0)

#     def get_precision(self):

#         prec = []
#         for i in range(self.n_classes):
#             prec.append(self.tp[i] / (self.tp[i]+ self.fp[i] + np.finfo(np.float64).eps))

#         return torch.mean(torch.stack(prec), 0)

#     def get_recall(self):

#         recall = []
#         for i in range(self.n_classes):
#             recall.append(self.tp[i] / (self.tp[i]+ self.fn[i] + np.finfo(np.float64).eps))

#         return torch.mean(torch.stack(recall), 0)


#     def get_f1(self):

#         f1 = []
#         for i in range(self.n_classes):
#             f1.append((2.0 * self.tp[i]) / (2.0 * self.tp[i] + self.fp[i] + self.fn[i]))

#         return torch.mean(torch.stack(f1), 0)

#     def get_der(self):

#         der = []

#         for i in range(self.n_classes):
#             der.append( (self.fp[i] + self.fn[i]) / (self.fn[i] + self.tp[i] + np.finfo(np.float64).eps))

#         return torch.mean(torch.stack(der), 0)

#     def get_matt(self):

#         matt = []

#         for i in range(self.n_classes):
#             matt.append((self.tp[i] * self.tn[i] - self.fp[i] * self.fn[i]) / \
#                (np.finfo(np.float64).eps + (self.tp[i] + self.fp[i]) * (self.tp[i] + self.fn[i]) * (self.tn[i] + self.fp[i]) * (
#                            self.tn[i] + self.fn[i])).sqrt()  )

#         return torch.mean(torch.stack(matt), 0)

#     def get_tot_examples(self):

#         tot = []
#         for i in range(self.n_classes):
#             tot.append( self.tp[i] + self.tn[i] + self.fp[i] + self.fn[i])

#         return torch.sum(torch.stack(tot), 0)

#     def get_positive_examples(self):
#         tot = []
#         for i in range(self.n_classes):
#             tot.append(self.tp[i] + self.fn[i])

#         return torch.sum(torch.stack(tot), 0)

#     def get_positive_examples_class(self, i):
#         return self.tp[i] + self.fn[i]
    
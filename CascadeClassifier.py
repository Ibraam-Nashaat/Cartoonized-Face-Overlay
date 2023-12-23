from skimage import io
import numpy as np
from AdaBoost.AdaBoost import *
import numpy as np
from utils import *
from haar_like_features import *
class CascadeClassifier():
    def __init__(self, overall_false_positive_rate, max_acceptable_false_positive_rate, min_acceptable_detection_rate, step = 0.05):
        """
        max_acceptable_false_positive_rate: maximum acceptable false positive rate for each layer
        min_acceptable_detection_rate: minimum acceptable detection rate for each layer
        overall_false_positive_rate: overall false positive rate for the cascade classifier
        clfs: list of strong classifiers
        """
        self.clfs = []
        self.thresholds = []

        self.f = max_acceptable_false_positive_rate
        self.d = min_acceptable_detection_rate
        self.F = overall_false_positive_rate
        self.step = step
        self.utils = Utils()


    def _train_classifier(self, P, N, n):
        """
        train an AdaBoost classifier with n features
        """
        clf = AdaBoostClassifier(n)
        X, y = self.utils.merge_P_N(P, N)
        clf.fit(X, y)
        return clf
    
    def _eval(self, clf, X_val, y_val, thresh=0.5, verboase=False):
        ypred = clf.predict_th(X_val, thresh)
        true_pos = np.sum((ypred == 1) & (y_val == 1))
        true_neg = np.sum((ypred == 0) & (y_val == 0))
        false_pos = np.sum((ypred == 1) & (y_val == 0))
        false_neg = np.sum((ypred == 0) & (y_val == 1))

        false_postive_rate = false_pos / (false_pos+true_neg)
        detection_rate = true_pos / (false_neg+true_pos)
        if verboase:
            print("\tTrue Positive: ", true_pos)
            print("\tTrue Negative: ", true_neg)
            print("\tFalse Positive: ", false_pos)
            print("\tFalse Negative: ", false_neg)
            print("\tFPR: ", false_postive_rate)
            print("\tDR: ", detection_rate)
        else:
            print("\tTP: ", true_pos, " TN: ", true_neg, " FP: ", false_pos, " FN: ", false_neg)
        
        return false_postive_rate, detection_rate
    
    def _update_N(self, N, clf, threshold):
        """
        update negative samples
        """
        false_N_pred = clf.predict_th(N, threshold)
        return N[false_N_pred == 1]
    
    def train(self, P_train, N_train, X_val, y_val):
            P = P_train
            N = N_train.copy()
            F1 = 1
            D1 = 1
            i = 0
            while F1 > self.F and len(N) > 0:
                i = i + 1
                F0 = F1
                D0 = D1
                n = 0
                print(f"=================== Training layer {i} FPR = {F1} targeting {self.f*F0} ====================")
                print(f"=================== Training layer {i} DR = {D1} targeting {self.d*D0} ====================")
                print(f"=================== Training layer {i} N: {N.shape}  ====================")
                while F1 > self.f * F0:
                    print(f"+ Start Training layer {i} with {n + 1} features ===")
                    n = n + 1
                    clf = self._train_classifier(P, N, n)
                    threshold = 1
                    F1, D1 = self._eval(clf, X_val, y_val, threshold)
                    
                    while D1 < self.d * D0:
                        print(f"- Revaluate Classifier with {threshold} threshold \n\t D = {D1} target_D = {self.d * D0} \n\t F = {F1} target_F = {self.f * F0}")
                        threshold -= self.step
                        if threshold < -1.0 : 
                            threshold = -1.0
                        F1, D1 = self._eval(clf, X_val, y_val, threshold)


                    print(f"=========================================================== ")
                    print(f"=========================================================== ")
                    print(f"Finished Training weak classifier with {n} features with: ")
                    print(f"=========================================================== ")
                    print(f"=========================================================== ")
                    print("\tFalse Postive rate = ", F1)
                    print("\tDetection rate = ", D1)
                    print("\tThreshold = ", threshold)
                    print("\t=====================")
                    self.clfs.append(clf)
                    self.utils.save_pickle((clf.models, clf.alphas, threshold), f"./models/layer_{i}_classifier_{n}.pkl")
                    self.thresholds.append(threshold)

                if F1 > self.F:
                    N = self._update_N(N, clf, threshold)
            
if __name__ == "__main__":
    classifier = CascadeClassifier(0.07, 0.60, 0.94)
    utils = Utils()

    X_train, y_train = utils.load_pickle('./dataset/train_dataset.pkl')
    X_val, y_val = utils.load_pickle('./dataset/val_dataset.pkl')   
    
    


    classifier.train(X_train[y_train == 1], X_train[y_train == 0], X_val, y_val)
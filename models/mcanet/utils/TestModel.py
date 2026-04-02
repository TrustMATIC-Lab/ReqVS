# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-03-29 14:00
LastEditTime: 2022-11-23 15:32
LastEditors: MrZQAQ
Description: Test model
FilePath: /MCANet/utils/TestModel.py
'''

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, f1_score)


def test_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):
    if isinstance(MODEL, list):
        for item in MODEL:
            item.eval()
    else:
        MODEL.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds, proteins, labels = data
            compounds = compounds.to(DEVICE)
            proteins = proteins.to(DEVICE)
            labels = labels.to(DEVICE)

            if isinstance(MODEL, list):
                predicted_scores = torch.zeros(2).to(DEVICE)
                for i in range(len(MODEL)):
                    predicted_scores = predicted_scores + \
                        MODEL[i](compounds, proteins)
                predicted_scores = predicted_scores / FOLD_NUM
            else:
                predicted_scores = MODEL(compounds, proteins)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(
                predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    return Y, P, test_loss, Accuracy, Precision, Recall, AUC, PRC


def test_model(MODEL, dataset_loader, save_path, DATASET, LOSS, DEVICE, dataset_class="Train", save=True, FOLD_NUM=1):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_loader)),
        total=len(dataset_loader))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_precess(
        MODEL, test_pbar, LOSS, DEVICE, FOLD_NUM)
    if save:
        if FOLD_NUM == 1:
            filepath = save_path + \
                "/{}_{}_prediction.txt".format(DATASET, dataset_class)
        else:
            filepath = save_path + \
                "/{}_{}_ensemble_prediction.txt".format(DATASET, dataset_class)
        with open(filepath, 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(dataset_class, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test


def test_model_outer(Model,test_file_path,dataset_loader,Device,FOLD_NUM=1):

    result_data= test_outer_process(
        Model, test_file_path, dataset_loader, Device, FOLD_NUM)

    return result_data


def test_outer_process(Model,test_file_path,dataset_loader,Device,FOLD_NUM):
    """
    remain the ID of DUD-E,
    the format of the data is ID SMILES Protein_sequence label
    need the output score of the model,which is the [:,1] of the softmax output
    return the whole data with ID,SMILES,Protein_sequence,label,score
    """
    Model.eval()
    True_labels = []
    Pred_scores = []
    result_data = []
    with torch.no_grad():
        for data in tqdm(dataset_loader, total=len(dataset_loader), desc="测试中"):
            ID,compounds,proteins,labels,orig_smiles,orig_protein = data
            compounds = compounds.to(Device)
            proteins = proteins.to(Device)
            labels = labels.to(Device)
            logits = Model(compounds,proteins)
            probs = F.softmax(logits,dim=1)
            pos_prob = probs[:,1].detach().cpu().numpy()

            True_labels.extend(labels.cpu().numpy())
            Pred_scores.extend(pos_prob)

            for j in range(len(ID)):
                result_data.append([
                    ID[j],
                    orig_smiles[j],
                    orig_protein[j],
                    int(labels[j]),
                    float(pos_prob[j])

                ])
    
    Y = np.array(True_labels)
    S = np.array(Pred_scores)
    ROC_AUC = roc_auc_score(Y, S)

    precision_curve, recall_curve, _ = precision_recall_curve(Y, S)
    PR_AUC = auc(recall_curve, precision_curve)



    metrics = {
        "ROC_AUC": ROC_AUC,
        "PR_AUC": PR_AUC,
    }
    print(f"Metrics: {metrics}")

    return result_data


            

def Virtual_Screen(Model,test_file_path,dataset_loader,Device,FOLD_NUM=1):

    result_data= Virtual_Screen_process(
        Model, test_file_path, dataset_loader, Device, FOLD_NUM)

    return result_data


def Virtual_Screen_process(Model,test_file_path,dataset_loader,Device,FOLD_NUM):
    """
    remain the ID of Virtual_Screen,
    the format of the data is ID SMILES Protein_sequence label
    need the output score of the model,which is the [:,1] of the softmax output
    return the whole data with ID,SMILES,Protein_sequence,label,score
    """
    Model.eval()
    True_labels = []
    Pred_scores = []
    result_data = []
    with torch.no_grad():
        for data in tqdm(dataset_loader, total=len(dataset_loader), desc="测试中"):
            ID,compounds,proteins,labels,orig_smiles,orig_protein = data
            compounds = compounds.to(Device)
            proteins = proteins.to(Device)
            # labels = labels.to(Device)
            logits = Model(compounds,proteins)
            probs = F.softmax(logits,dim=1)
            pos_prob = probs[:,1].detach().cpu().numpy()

            # True_labels.extend(labels.cpu().numpy())
            # Pred_scores.extend(pos_prob)

            for j in range(len(ID)):
                result_data.append([
                    ID[j],
                    orig_smiles[j],
                    # orig_protein[j],
                    # int(labels[j]),
                    float(pos_prob[j])

                ])
    
    # Y = np.array(True_labels)
    # S = np.array(Pred_scores)
    # ROC_AUC = roc_auc_score(Y, S)

    # precision_curve, recall_curve, _ = precision_recall_curve(Y, S)
    # PR_AUC = auc(recall_curve, precision_curve)



    # metrics = {
    #     "ROC_AUC": ROC_AUC,
    #     "PR_AUC": PR_AUC,
    # }
    # print(f"Metrics: {metrics}")

    return result_data
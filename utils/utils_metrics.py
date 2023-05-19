import csv
import os
import sys
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def InitList(x, y):
    array = [([0] * y) for i in range(x)]
    return array


def GetListByCoord(array, array2, radius, x, y):

    row_col = 2 * radius + 1

    result = InitList(row_col, row_col)

    arrayRow, arrayCol = len(array), len(array[0])

    for i in range(result.__len__()):
        for j in range(result.__len__()):

            if (i + x - radius < 0 or j + y - radius < 0 or i + x - radius >= arrayRow or j + y - radius >= arrayCol):
                continue
            elif (array[x, y] == array2[i + x - radius, j + y - radius]):
                return 1
    return 0


def canny(filename):

    binary1 = filename
    binary1 = binary1.astype(np.uint8)

    binary = cv2.Canny(binary1, 1, 1 * 3, apertureSize=3)

    return binary
def contours(img):
    
    mask = np.zeros(img.shape, np.uint8)
    if img.max()==2:
        for i in range(2):
            if i==0:
                sub_img = np.where(img==2,0,img)
                contours, hierarchy = cv2.findContours(sub_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(mask, contours, -1, 1, 1)  
            else:
                sub_img = np.where(img==1,0,img)
                contours, hierarchy = cv2.findContours(sub_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(mask, contours, -1, 1, 1) 
    else:
        sub_img = img
        contours, hierarchy = cv2.findContours(sub_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) 
        cv2.drawContours(mask, contours, -1, 1, 1)  

    

    return mask
def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

def Frequency_Weighted_Intersection_over_Union(confusion_matrix):

    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def fast_hist(a, b, n):

    k = (a >= 0) & (a < n)

    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 
def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
    print('Num classes', num_classes)  

    hist = np.zeros((num_classes, num_classes))
    
   
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]  

    for ind in range(len(gt_imgs)): 

        pred = np.array(Image.open(pred_imgs[ind]))  

        label = np.array(Image.open(gt_imgs[ind]))  

        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  

        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )

    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)

    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))


    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))  
    return np.array(hist, np.int), IoUs, PA_Recall, Precision
def compute_mIoU_(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
    print('Num classes', num_classes)  

    hist = np.zeros((num_classes, num_classes))

    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]
    sumk = 0
    sumk1 = 0
    sumindex = 0
    sumindex1 = 0

    for ind in range(len(gt_imgs)): 

        pred = np.array(Image.open(pred_imgs[ind]))  

        label = np.array(Image.open(gt_imgs[ind]))  

        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue


        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )
    for ind in range(len(gt_imgs)):

        y_predict1 = cv2.imread(pred_imgs[ind], -1)
        labell = cv2.imread(gt_imgs[ind], -1)  
        y_predict1 = contours(y_predict1)
        labell = contours(labell)

        index = np.argwhere(y_predict1 != 0)  
        index1 = np.argwhere(labell != 0)  
        k = 0
        k1 = 0
        for i in range(index.shape[0]):
            x = index[i][0]
            y = index[i][1]
            k += GetListByCoord(y_predict1, labell, 1, x, y) 
        for ii in range(index1.shape[0]):
            x = index1[ii][0]
            y = index1[ii][1]
            k1 += GetListByCoord(labell, y_predict1, 1, x, y)

        sumk += k
        sumk1 += k1
        sumindex += index.shape[0]
        sumindex1 += index1.shape[0]

    line_BP = sumk / sumindex

    line_BR = sumk1 / sumindex1

    line_F1 = (2 * line_BP * line_BR) / (line_BR + line_BP)

    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)

    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))


    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))  
    f1_score = 2*PA_Recall*Precision/(PA_Recall+Precision)
    FWMIOU = Frequency_Weighted_Intersection_over_Union(hist)
    matricData = [
        {"class": "background", "iou": round(IoUs[0], 4), "precision": round(Precision[0], 4), "recall": round(PA_Recall[0], 4), "f1-score": round(f1_score[0], 4), "MIoU": round(np.nanmean(IoUs), 4), "MPA": round(np.nanmean(PA_Recall) , 4), "FWMIoU": round(FWMIOU, 4),
         "boundary_BP": round(line_BP, 4), "boundary_BR": round(line_BR, 4), "boundary_F1": round(line_F1, 4)},
        {"class": "runway", "iou": round(IoUs[1], 4), "precision": round(Precision[1], 4), "recall": round(PA_Recall[1], 4), "f1-score": round(f1_score[1], 4), "MIoU": "", "MPA": "", "FWMIoU": "",
         "boundary_BP": "", "boundary_BR": "", "boundary_F1": ""},
        {"class": "liaison", "iou": round(IoUs[2], 4), "precision": round(Precision[2], 4), "recall": round(PA_Recall[2], 4), "f1-score": round(f1_score[2], 4), "MIoU": "", "MPA": "", "FWMIoU": "",
         "boundary_BP": "", "boundary_BR": "", "boundary_F1": ""},
    ]


    return np.array(hist, np.int), IoUs, PA_Recall, Precision,f1_score, round(np.nanmean(IoUs) * 100, 2),round(np.nanmean(PA_Recall) * 100, 2), FWMIOU,matricData

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))

import xlsxwriter as xw
def xw_toExcel(data, fileName): 
    workbook = xw.Workbook(fileName)  
    worksheet1 = workbook.add_worksheet("sheet1")  
    worksheet1.activate()  
    title = ['classes', 'iou', 'precision', 'recall', 'f1-score', 'MIoU', 'MPA', 'FWMIoU','boundary_BP', 'boundary_BR', 'boundary_F1']  # 设置表头
    worksheet1.write_row('A1', title) 
    i = 2  
    for j in range(len(data)):
        insertData = [data[j]["class"], data[j]["iou"], data[j]["precision"], data[j]["recall"], data[j]["f1-score"], data[j]["MIoU"], data[j]["MPA"],data[j]["FWMIoU"], data[j]["boundary_BP"], data[j]["boundary_BR"], data[j]["boundary_F1"]]
        row = 'A' + str(i)
        worksheet1.write_row(row, insertData)
        i += 1
    workbook.close()  
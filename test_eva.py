import sys
import time
import os
import glob
import glob
import os
import sys
import time
from utils.utils_metrics import compute_mIoU_,xw_toExcel
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from nets.model import DCBP
colors = [(0, 0, 0), (0, 0, 128), (0, 128, 0)]
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
def init(model_path='', num_classes=3, backbone='hrnetv2_w18', cuda=True, onnx=False):
    net = DCBP(num_classes=num_classes, backbone=backbone, pretrained=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.eval()
    print('{} model, and classes loaded.'.format(model_path))
    if not onnx:
        if cuda:
            net = nn.DataParallel(net)
            net = net.cuda()
    return net


def process_image(handle=None, input_image=None, img_size=(512, 512), output_path='',line_output_path=''):
    img_size = img_size
    model = handle
    h, w, _ = input_image.shape
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(input_image, img_size, cv2.INTER_LINEAR)
    img = np.array(img, np.float32)
    img /= 255.0
    img = np.expand_dims(np.transpose(np.array(img, np.float32), (2, 0, 1)), 0)
    img = torch.from_numpy(img)
    img = img.cuda()
    with torch.no_grad():
        output = model(img)[0][0]
        output = F.softmax(output.permute(1, 2, 0), dim=-1).cpu().numpy()
    
    output = np.argmax(output, axis=-1)
    output = np.squeeze(output).astype(np.uint8)
    y_pre = cv2.resize(output, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(output_path, y_pre)
    return y_pre

def get_image_names(pred_dir):
    pred_paths = glob.glob(os.path.join(pred_dir, "*.png"))
    name_list = []
    for i in range(len(pred_paths)):
        img_path = pred_paths[i]
        if 'win' in sys.platform:
            name = img_path.split('.')[0].split('\\')[-1]
        else:
            name = img_path.split('.')[0].split('/')[-1]
        name_list.append(name)
    return name_list
if __name__ == "__main__":
    model_paths = "/data2/jw/runway_seg/DCBP/model_data/best.pth"
    out_path = "/data2/jw/runway_seg/DCBP/results/mm"
    gt_dir = '/data2/jw/runway_seg/dataset/runway/labels'
    num_classes = 3
    backbone = "hrnetv2_w32"
    net = init(model_paths, num_classes, backbone, cuda=True, onnx=False)
    input_shape = [512, 512]
    img_paths = glob.glob(os.path.join("/data2/jw/runway_seg/dataset/runway/test", "*.png"))
    render_map_path = os.path.join(out_path,'render_map')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(render_map_path):
        os.makedirs(render_map_path)
    for i in range(len(img_paths)):
        img = cv2.imread(img_paths[i])
        orininal_h, orininal_w, _ = img.shape
        if 'win' in sys.platform:
            name = img_paths[i].split('.')[0].split('\\')[-1]
        else:
            name = img_paths[i].split('.')[0].split('/')[-1]
        output_path = os.path.join(out_path, name + ".png")
        line_output_path = os.path.join(out_path, name + "_boudary.png")
        output_render_map_path = os.path.join(render_map_path, name + ".png")
        line_output_render_map_path = os.path.join(render_map_path, name +"_boudary.png")
        y_pre = process_image(net, img, input_shape, output_path=output_path,line_output_path=line_output_path)
        seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(y_pre, [-1])], [orininal_h, orininal_w, -1])
        image = Image.fromarray(np.uint8(seg_img))
        old_img = Image.fromarray(np.uint8(img))
        render_map = seg_img * 0.8 + img
        cv2.imwrite(output_render_map_path, render_map)
        
    pred_dir = out_path
    image_names = get_image_names(pred_dir)
    miou_out_path = os.path.join(pred_dir,'miou')
    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)
    num_classes = 3
    name_classes = ["background" ,"runway", "liaison"]
    hist, IoUs, PA_Recall, Precision, F1, MIoU, MPA, FWMIoU, matricData = compute_mIoU_(gt_dir, pred_dir, image_names, num_classes, name_classes)
    print(matricData)
    if not os.path.exists(os.path.join(pred_dir,"matrics")):
        os.makedirs(os.path.join(pred_dir,"matrics"))
    xw_toExcel(matricData,os.path.join(pred_dir,"matrics","matrics.xlsx"))
import torch
import random
import numpy as np
import argparse
import os
import cv2
import json
import feather
import pandas as pd
import matplotlib.pyplot as plt

from parsing.utils.comm import to_device
from parsing.dataset import build_test_dataset
from decoder import EvRoomDetector
from utils.tester import eval_sap
from utils.tester import run_nms
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

def event2img(events):
    room1 = []
    room2 = []
    for i in range(len(events)):
        if events[i][3] == 1:
            room1.append([events[i][1],  events[i][2]])
        if events[i][3] == 0:
            room2.append([events[i][1], events[i][2]])
    image = np.ones((720, 1280, 3), np.uint8) * 255
    points_list_1 = room1
    points_list_2 = room2
    for point in points_list_1:
        cv2.circle(image, point, 1, (255, 0, 0), -1)
    for point in points_list_2:
        cv2.circle(image, point, 1, (0, 0, 255), -1)
    output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return output_image


def Inference(model_path, cuda, data_dir):
    device = cuda
    model = EvRoomDetector().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    inference_dir = data_dir

    npy_files = [f for f in os.listdir(inference_dir+ '/voxel_grid/') if f.endswith('.npy')]

    result = {}
    for filename in npy_files:
        vg = np.load(inference_dir + '/voxel_grid/' + filename)
        vg = vg.transpose((2, 0, 1))
        vg = vg[3:, :, :]
        image = {}
        image['img'] = torch.tensor(vg).unsqueeze(0).float()
        map_oirg = np.load(inference_dir + '/orig_map/' + filename)
        map_shift = np.load(inference_dir + '/shift_map/' + filename)
        image['map_orig'] = torch.tensor(map_oirg).unsqueeze(0)
        image['map_shift'] = torch.tensor(map_shift).unsqueeze(0)
        image = to_device(image, device)

        ann = {
            'height': vg.shape[1],
            'width': vg.shape[2],
            'filename': filename
        }
        ann = to_device(ann, device)
        with torch.no_grad():
            lines_final, line_logits, juncs_final, juncs_logits, output, extra_info = \
                                model.forward_test(image, [ann])
            output = to_device(output, 'cpu')
            run_nms(output)
            for k in output.keys():
                if isinstance(output[k], torch.Tensor):
                    output[k] = output[k].tolist()
            result[filename] = output

        with open(inference_dir + filename.split('.')[0] +'_result.json','w') as _out:
            json.dump(result,_out)

        scores = np.array(output['lines_valid_score'])
        indices = np.where(scores > 0.5)[0]
        lines = np.array(output['lines_pred'])[indices]
        label = np.array(output['lines_label'])[indices]

        events = pd.read_feather(
            inference_dir + '/events/' + filename.split('.')[0] + '.feather').to_numpy()
        events = events[:, :]
        event_img = event2img(events)
        plt.imshow(event_img, cmap='gray')
        colors = ['black', '#ff531a', '#082464', '#f0b404', '#2eb82e', '#10ecdc']

        for i, l in enumerate(lines):
            x1, y1, x2, y2 = l
            x1 = int(x1 * 1280 / 256)
            x2 = int(x2 * 1280 / 256)
            y1 = int(y1 * 720 / 256)
            y2 = int(y2 * 720 / 256)
            category = label[i]
            color = colors[category]
            plt.plot([x1, x2], [y1, y2], color=color, linewidth=8)
        plt.axis('off')
        plt.savefig(inference_dir + filename.split('.')[0] + '_visual_result.png', bbox_inches='tight')
        plt.close()
        # plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Ev-Layout Inference')
    parser.add_argument("--model_path", default='./model/EV-Layout.pth', type=str)
    parser.add_argument("--cuda_id", default='cuda:2', type=str)
    parser.add_argument("--data_dir", default='./inference_data/', type=str)
    parser.add_argument("--seed", default=2, type=int)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    Inference(args.model_path, args.cuda_id, args.data_dir)

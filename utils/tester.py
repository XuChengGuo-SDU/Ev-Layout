import utils.metric_evaluation as me
from utils.labels import LabelMapper
import os.path as osp
from tabulate import tabulate
import torch
import matplotlib.pyplot as plt
from utils.visualization import ImagePlotter
import numpy as np

LINE_LABELS = ['invalid', 'wall', 'floor', 'ceiling', 'window', 'door']
JUNCTION_LABELS = ['invalid', 'false', 'proper']
DISABLE_CLASSES = False

lm = LabelMapper(LINE_LABELS, JUNCTION_LABELS, disable=DISABLE_CLASSES)
img_viz = ImagePlotter(lm.get_line_labels(), lm.get_junction_labels())
plot_dir = './outputs'

def eval_sap(results, annotations_dict, epoch):
    thresholds = [5, 10, 15]  # <<<<<
    rcs, pcs, sAP = me.evalulate_sap(results, annotations_dict, thresholds, lm.get_line_labels())
    for m_type, thres_dict in sAP.items():
        for t in thres_dict:
            try:
                fig = img_viz.plot_ap(rcs[m_type][t], pcs[m_type][t], sAP[m_type][t], t,
                                           AP_string=fr'\mathrm{{sAP}}')
                fig_path = osp.join(plot_dir, 'E{:02}_sAP_{}_{}.pdf'.format(epoch, m_type, t))
                plt.savefig(fig_path)
            except KeyError:
                fig = None
    thresholds = [0.5, 1.0, 2.0]
    rcs, pcs, jAP = me.evalulate_jap(results, annotations_dict, thresholds, lm.get_junction_labels())
    ap_str = {'valid': r'\mathrm{{j}}_1\mathrm{{AP}}',
              'label': r'\mathrm{{j}}_3\mathrm{{AP}}',
              'label_line_valid': r'\mathrm{{j}}_2\mathrm{{AP}}'}
    for m_type, thres_dict in jAP.items():
        dstr = ap_str[m_type]
        for t in thres_dict:
            try:
                fig = img_viz.plot_ap(rcs[m_type][t], pcs[m_type][t], jAP[m_type][t], t, AP_string=dstr)
                fig_path = osp.join(plot_dir, 'E{:02}_jAP_{}_{}.pdf'.format(epoch, m_type, t))
                plt.savefig(fig_path)
            except KeyError:
                fig = None
    return sAP, jAP

def run_nms(output_list, ax=None):
    line_nms_threshold2 = 9
    graph_nms = None
    if line_nms_threshold2 <= 0 and not graph_nms:
        return
    if hasattr(output_list, 'keys'):
        output_list = [output_list]

    for output in output_list:
        if line_nms_threshold2 > 0:
            _nms_distance(output)


def _nms_distance(output):
    line_nms_threshold2 = 9

    line_segments = np.array(output['lines_pred'], dtype=np.float32)
    if line_segments.shape[0] < 2:
        return
    line_segments[:, 0] *= 64 / float(output['width'])
    line_segments[:, 1] *= 64 / float(output['height'])
    line_segments[:, 2] *= 64 / float(output['width'])
    line_segments[:, 3] *= 64 / float(output['height'])
    line_segments = line_segments.reshape(-1, 2, 2)[:, :, ::-1]
    score = np.array(output['lines_label_score'], dtype=np.float32)
    labels = np.array(output['lines_label'], dtype=int)
    remove_mask = np.zeros(line_segments.shape[0], dtype=bool)
    for label_idx in range(1, lm.nbr_line_labels()):
        label_mask = labels == label_idx
        global_idx = np.flatnonzero(label_mask)
        label_ls = line_segments[label_mask]
        label_score = score[label_mask]

        diff = ((label_ls[:, None, :, None] - label_ls[:, None]) ** 2).sum(-1)
        diff = np.minimum(
            diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
        )
        diff[np.arange(label_ls.shape[0]), np.arange(label_ls.shape[0])] = np.inf
        for g_idx, l_diff in zip(global_idx, diff):
            matches_score = label_score[l_diff < line_nms_threshold2]
            if matches_score.size > 0:
                remove_mask[g_idx] |= np.any(matches_score > score[g_idx])

    if np.any(remove_mask):
        _nms_remove(remove_mask, output)

def _nms_remove(remove_mask, output):
    if isinstance(output['lines_pred'], torch.Tensor):
        t_mask = torch.tensor(~remove_mask)
        output['lines_pred'] = output['lines_pred'][t_mask]
        output['lines_label_score'] = output['lines_label_score'][t_mask]
        output['lines_valid_score'] = output['lines_valid_score'][t_mask]
        output['lines_label'] = output['lines_label'][t_mask]
        output['lines_score'] = output['lines_score'][t_mask]
        if 'line2junc_idx' in output:
            output['line2junc_idx'] = output['line2junc_idx'][t_mask]
    else:
        keep_idx = np.flatnonzero(~remove_mask)
        output['lines_pred'] = [output['lines_pred'][idx] for idx in keep_idx]
        output['lines_label_score'] = [output['lines_label_score'][idx] for idx in keep_idx]
        output['lines_valid_score'] = [output['lines_valid_score'][idx] for idx in keep_idx]
        output['lines_label'] = [output['lines_label'][idx] for idx in keep_idx]
        output['lines_score'] = [output['lines_score'][idx] for idx in keep_idx]
        if 'line2junc_idx' in output:
            output['line2junc_idx'] = [output['line2junc_idx'][idx] for idx in keep_idx]


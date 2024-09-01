import os
import time
import string
import argparse
import re
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk_bleu import doc_bleu
from utils import Averager
from model import make_std_mask

from contrastive_loss import contrastive_loss_cal_mcl

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validation_modal_contrastive_learning_timt_task(model_list, criterion, evaluation_loader, src_converter, tgt_converter, opt):
    """ validation or evaluation """
    n_correct = 0
    n_total = 0
    norm_ED = 0
    ref_sents = []
    pred_sents = []
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    valid_loss_before = valid_loss_avg.val()
    
    for i, (image_tensors, _, _, src_labels, tgt_labels, _, _) in enumerate(evaluation_loader):   
        print('Decoding batch {} in validation ...'.format(i+1))
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        src_length_for_pred = torch.IntTensor([opt.src_batch_max_length] * batch_size).to(device)
        tgt_length_for_pred = torch.IntTensor([opt.tgt_batch_max_length] * batch_size).to(device)
        length_for_pred = tgt_length_for_pred
        src_text_for_pred = torch.LongTensor(batch_size, opt.src_batch_max_length + 1).fill_(0).to(device)
        tgt_text_for_pred = torch.LongTensor(batch_size, opt.tgt_batch_max_length + 1).fill_(0).to(device)
        
        src_text_for_loss, src_length_for_loss = src_converter.encode(src_labels, opt.src_level, batch_max_length=opt.src_batch_max_length)
        tgt_text_for_loss, tgt_length_for_loss = tgt_converter.encode(tgt_labels, opt.tgt_level, batch_max_length=opt.tgt_batch_max_length)
        text_for_loss = tgt_text_for_loss
        length_for_loss = tgt_length_for_loss
        valid_tgt_mask = make_std_mask(tgt_text_for_loss[:, :-1], pad = 2)
        valid_src_mask = opt.src_mask
        valid_tgt_mask = opt.tgt_mask

        if opt.num_gpu > 1:
            x_tgt_mask, y_tgt_mask = valid_tgt_mask.size()
            new_tgt_mask = valid_tgt_mask.repeat(opt.batch_size, 1)
            new_tgt_mask = new_tgt_mask.reshape(opt.batch_size, x_tgt_mask, y_tgt_mask)
            valid_tgt_mask = new_tgt_mask
        
        start_time = time.time()
        start_symbol = 0
        src_preds = src_text_for_pred
        tgt_preds = tgt_text_for_pred
        src_decoder_input = src_text_for_pred
        tgt_decoder_input = tgt_text_for_pred
        timt_decoder_input = tgt_text_for_pred
        ocr_decoder_input = src_text_for_pred
        mt_decoder_input = tgt_text_for_pred

        for i in range(opt.tgt_batch_max_length + 1):
            visual_feature = model_list[0](input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
            textual_feature = model_list[1](input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
            visual_contextual_feature = model_list[2](visual_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
            textual_contextual_feature = model_list[2](textual_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
            
            timt_preds = model_list[3](contextual_feature = visual_contextual_feature, input = image, text = timt_decoder_input.long(), tgt_mask = valid_tgt_mask, is_train=False)
            mt_preds = model_list[3](contextual_feature = textual_contextual_feature, input = image, text = mt_decoder_input.long(), tgt_mask = valid_tgt_mask, is_train=False)
            
            _, timt_preds_index = timt_preds.max(2)
            _, mt_preds_index = mt_preds.max(2)
            
            if i+1 < opt.tgt_batch_max_length + 1:
                timt_decoder_input[:, i+1] = timt_preds_index[:, i]
                mt_decoder_input[:, i+1] = mt_preds_index[:, i]

        forward_time = time.time() - start_time
        
        timt_preds = timt_preds[:, :tgt_text_for_loss.shape[1] - 1, :]
        mt_preds = mt_preds[:, :tgt_text_for_loss.shape[1] - 1, :]
        
        src_target = src_text_for_loss[:, 1:]  # without [GO] Symbol
        tgt_target = tgt_text_for_loss[:, 1:]  # without [GO] Symbol
        timt_cost = criterion(timt_preds.contiguous().view(-1, timt_preds.shape[-1]), tgt_target.contiguous().view(-1))
        mt_cost = criterion(mt_preds.contiguous().view(-1, mt_preds.shape[-1]), tgt_target.contiguous().view(-1))
        if opt.CL_Task:
            pass
            fv = torch.sum(visual_contextual_feature, dim = 1) / visual_contextual_feature.shape[1]
            ft = torch.sum(textual_contextual_feature, dim = 1) / textual_contextual_feature.shape[1]
            
            if opt.CL_type == 'mcl':
                cl_cost = contrastive_loss_cal_mcl(fv, ft, opt.CL_tau)
            else:
                print('No clear contrastive loss type defined. Set cl_cost to 0!!!')
                cl_cost = 0

        # select max probabilty (greedy decoding) then decode index to character
        _, timt_preds_index = timt_preds.max(2)
        timt_preds_str = tgt_converter.decode(timt_preds_index, tgt_length_for_pred, opt.tgt_level)
        
        _, mt_preds_index = mt_preds.max(2)
        mt_preds_str = tgt_converter.decode(mt_preds_index, tgt_length_for_pred, opt.tgt_level)
        src_labels = src_converter.decode(src_text_for_loss[:, 1:], src_length_for_loss, opt.src_level)
        tgt_labels = tgt_converter.decode(tgt_text_for_loss[:, 1:], tgt_length_for_loss, opt.tgt_level)
        
        infer_time += forward_time
        if opt.CL_Task:
            if opt.MT_Weight > -1:
                MT_Weight = opt.MT_Weight
                CL_Weight = opt.CL_Weight
                TIMT_Weight = 1 - MT_Weight - CL_Weight
            else:
                print('Please input correct loss weights of machine translation task.')
                exit()
            
            if opt.MT_Weight >= 0.9999:
                MT_Weight = opt.MT_Weight
                CL_Weight = opt.CL_Weight
                TIMT_Weight = 1
            
            weighted_timt_cost = TIMT_Weight * timt_cost
            weighted_mt_cost = MT_Weight * mt_cost
            weighted_cl_cost = CL_Weight * cl_cost
            
            valid_loss_avg.add(weighted_timt_cost)
            valid_loss_avg.add(weighted_mt_cost)
            valid_loss_avg.add(weighted_cl_cost)
        else:
            print('Please input correct state of opt.cl(yes or no) | whether use contrastive learning or not.')
            exit()
        
        for gt, pred in zip(tgt_labels, timt_preds_str):
            gt = gt[:gt.find('[s]')]
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

            ref_sents.append(gt)
            pred_sents.append(pred)
            for item in gt:
                n_total += 1
                if item in pred:
                    n_correct += 1

    accuracy = n_correct / float(n_total) * 100
    tok_bleu, char_bleu = doc_bleu(ref_sents, pred_sents)
    return valid_loss_avg.val(), accuracy, tok_bleu, timt_preds_str, mt_preds_str, src_labels, tgt_labels, infer_time, length_of_data




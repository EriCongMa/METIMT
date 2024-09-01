# coding: utf-8
import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import AttnLabelConverter, Averager

from contrastive_loss import contrastive_loss_cal_mcl

from dataset import (
    TextualPairDataset, Batch_Balanced_Dataset_otmi_7, hierarchical_dataset_otmi_7, AlignCollate_otmi_7
    )
from model import (
    make_std_mask, Visual_Encoder, Textual_Encoder, Transformer_Encoder, Transformer_Decoder
)
from validate import (
    validation_modal_contrastive_learning_timt_task
)


import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print('Stopped! Because Only CPU could be available ...')
    exit()
else:
    print('Now is using device: {}'.format(device))


def modal_contrastive_learning_timt_train(opt):
    print('Load task modal_contrastive_learning_timt_train successfully.')
    
    # Split dataset if there are multiple datasets are used.
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    print('-' * 80)

    # Loading Training Dataset
    train_dataset = Batch_Balanced_Dataset_otmi_7(opt)
    print('Length of train_dataset: {}'.format(len(train_dataset)))
    print('-' * 80)

    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    ## This part is used to load textual parallel data
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    # Load textual parallel data
    if opt.external_mt == 'yes':
        print('-' * 80)
        print('Loading textual parallel data ...')
        text_train_data = TextualPairDataset(opt.src_train_text, opt.tgt_train_text, opt)
        text_train_dataset = torch.utils.data.DataLoader(
            text_train_data, batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.workers))
        text_train_loader = iter(text_train_dataset)

        # Methods to load textual data mini-batch:
        epoch_num = 0
        try:
            text_src_labels, text_tgt_labels = text_train_loader.next()
            print('Now in epoch {}.'.format(epoch_num))
        except:
            print('Start a new epoch!')
            epoch_num += 1
            text_train_loader = iter(text_train_dataset)
            text_src_labels, text_tgt_labels = text_train_loader.next()
            print('Now in epoch {}.'.format(epoch_num))

        assert len(text_src_labels) == len(text_tgt_labels)
        print('Length of text_train_dataset: {}'.format(len(text_train_data)))
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    
    print('Finished Loading Training Set.')

    print('-' * 80)
    print('Use hierarchical_dataset_otmi to load Valid Dataset ...')

    # Loading Validation Dataset
    AlignCollate_valid = AlignCollate_otmi_7(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset_otmi_7(root=opt.valid_data, opt=opt)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    print(valid_dataset_log)
    print('Length of valid_dataset: {}'.format(len(valid_dataset)))
    print('-' * 80)
    
    print('Length of train_dataset: {}'.format(len(train_dataset)))
    print('Length of valid_dataset: {}'.format(len(valid_dataset)))
    
    print('Finished Loading Training and validation Data!')
    
    """ model configuration """
    print('-' * 80)
    print('Now in model configuration')   
    src_converter = AttnLabelConverter(opt.src_character)
    tgt_converter = AttnLabelConverter(opt.tgt_character)

    opt.src_num_class = len(src_converter.character)
    opt.tgt_num_class = len(tgt_converter.character)

    if opt.rgb:
        opt.input_channel = 3
    
    # Construct Model information 
    ################################################################################
    # Define model modules
    ##### ##### ##### ##### ##### ##### #####
    # End-to-end TIMT Part
    visual_encoder = Visual_Encoder(opt)
    encoder = Transformer_Encoder(opt)
    tgt_decoder = Transformer_Decoder(opt, opt_dim = opt.tgt_num_class)
    # External MT Textual Encoder Part
    textual_encoder = Textual_Encoder(opt)

    model_list = [visual_encoder, textual_encoder, encoder, tgt_decoder]
    model_name_list = ['visual_encoder', 'textual_encoder', 'encoder', 'tgt_decoder']

    # weight initialization
    print('-' * 80)
    print('Now in weight initialization')
    for sub_model in model_name_list:
        for name, param in eval(sub_model).named_parameters():
            if 'localization_fc2' in name:
                continue
            try:
                if 'Transformer_encoder_layer' in name or 'Transformer_decoder_layer' in name \
                    or 'TransformerDecoder' in name or 'SequenceModeling' in name:
                    if param.dim() > 1:
                        init.xavier_uniform_(param)
                        continue
            except:
                pass
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue

    # Settings for multi-GPU
    visual_encoder = torch.nn.DataParallel(visual_encoder).to(device)
    textual_encoder = torch.nn.DataParallel(textual_encoder).to(device)
    encoder = torch.nn.DataParallel(encoder).to(device)
    tgt_decoder = torch.nn.DataParallel(tgt_decoder).to(device)
    
    visual_encoder.train()
    textual_encoder.train()
    encoder.train()
    tgt_decoder.train()
    
    # Setup Criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    
    # Setup Loss Averager
    loss_avg = Averager()
    timt_loss_avg = Averager()
    mt_loss_avg = Averager()
    if opt.CL_Task:
        cl_loss_avg = Averager()
    
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    ## Textual Loading Part
    ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
    if opt.external_mt == 'yes':
        external_mt_loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for sub_model in model_name_list:
        for p in filter(lambda p: p.requires_grad, eval(sub_model).parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
    print('Trainable params num : {}'.format(sum(params_num)))

    # setup optimizer
    print('-' * 80)
    print('Now in setup optimizer')
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)

    # Start Training 
    print('-' * 80)
    print('Start Training')
    start_iter = 0

    start_time = time.time()
    best_accuracy = -1
    best_bleu = -1
    best_valid_loss = 1000000
    iteration = start_iter - 1
    previous_best_accuracy_iter = 0
    previous_best_bleu_iter = 0
    previous_best_valid_iter = 0
    
    old_time = time.time()

    while(True):
        iteration += 1
        image_tensors_1, image_tensors_2, image_tensors_3, src_labels, tgt_labels, src_labels_teacher, tgt_labels_teacher = train_dataset.get_batch()        
        
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        ## Textual Loading Part
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        if opt.external_mt == 'yes':
            try:
                text_src_labels, text_tgt_labels = text_train_loader.next()
                print('Overall Step: {}. Textual Parallel Data: Now in epoch {}.'.format(iteration+1, epoch_num))
            except:
                print('Textual Parallel Data: Start a new epoch!')
                text_train_loader = iter(text_train_dataset)
                text_src_labels, text_tgt_labels = text_train_loader.next()
                epoch_num += 1
                print('Overall Step: {}. Textual Parallel Data: Now in epoch {}.'.format(iteration+1, epoch_num))
        
        image_1 = image_tensors_1.to(device)
        image_2 = image_tensors_2.to(device)
        image_3 = image_tensors_3.to(device)

        image = image_1
        
        # Textual data transformation: For triple img-src-tgt
        src_text, src_length = src_converter.encode(src_labels, opt.src_level,  batch_max_length=opt.src_batch_max_length)
        tgt_text, tgt_length = tgt_converter.encode(tgt_labels, opt.tgt_level,  batch_max_length=opt.tgt_batch_max_length)

        src_text_teacher, src_length_teacher = src_converter.encode(src_labels_teacher, opt.src_level,  batch_max_length=opt.src_batch_max_length)
        tgt_text_teacher, tgt_length_teacher = tgt_converter.encode(tgt_labels_teacher, opt.tgt_level,  batch_max_length=opt.tgt_batch_max_length)
        
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        ## Textual Loading Part
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # Textual data transformation: For textual parallel
        if opt.external_mt == 'yes':
            textual_src_text, textual_src_length = src_converter.encode(text_src_labels, opt.src_level,  batch_max_length=opt.src_batch_max_length)
            textual_tgt_text, textual_tgt_length = tgt_converter.encode(text_tgt_labels, opt.tgt_level,  batch_max_length=opt.tgt_batch_max_length)

        batch_size = image.size(0)
        text = tgt_text
        length = tgt_length

        print('batch_size is: {}'.format(batch_size))

        src_mask = make_std_mask(src_text[:, :-1], pad = 2)[0]
        tgt_mask = make_std_mask(tgt_text[:, :-1], pad = 2)[0]
        opt.src_mask = src_mask
        opt.tgt_mask = tgt_mask

        if opt.num_gpu > 1:
            x_tgt_mask, y_tgt_mask = tgt_mask.size()
            new_tgt_mask = tgt_mask.repeat(opt.batch_size, 1)
            new_tgt_mask = new_tgt_mask.reshape(opt.batch_size, x_tgt_mask, y_tgt_mask)
            tgt_mask = new_tgt_mask
        
        ###### Triple img-src-tgt side data forward ...
        visual_feature_1 = visual_encoder(input = image_1, text = src_text[:, :-1], tgt_mask = tgt_mask)
        visual_feature_2 = visual_encoder(input = image_2, text = src_text[:, :-1], tgt_mask = tgt_mask)
        visual_feature_3 = visual_encoder(input = image_3, text = src_text[:, :-1], tgt_mask = tgt_mask)

        textual_feature = textual_encoder(input = image, text = src_text[:, :-1], tgt_mask = tgt_mask)
        textual_feature_teacher = textual_encoder(input = image, text = src_text_teacher[:, :-1], tgt_mask = tgt_mask)

        visual_contextual_feature_1 = encoder(visual_feature_1, input = image_1, text = src_text[:, :-1], tgt_mask = tgt_mask)
        visual_contextual_feature_2 = encoder(visual_feature_2, input = image_2, text = src_text[:, :-1], tgt_mask = tgt_mask)
        visual_contextual_feature_3 = encoder(visual_feature_3, input = image_3, text = src_text[:, :-1], tgt_mask = tgt_mask)

        textual_contextual_feature = encoder(textual_feature, input = image, text = src_text[:, :-1], tgt_mask = tgt_mask)
        textual_contextual_feature_teacher = encoder(textual_feature_teacher, input = image, text = src_text_teacher[:, :-1], tgt_mask = tgt_mask)

        # print('run timt_preds ...')
        timt_preds_1 = tgt_decoder(contextual_feature = visual_contextual_feature_1, input = image_1, text = tgt_text[:, :-1], tgt_mask = tgt_mask)
        timt_preds_2 = tgt_decoder(contextual_feature = visual_contextual_feature_2, input = image_2, text = tgt_text[:, :-1], tgt_mask = tgt_mask)
        timt_preds_3 = tgt_decoder(contextual_feature = visual_contextual_feature_3, input = image_3, text = tgt_text[:, :-1], tgt_mask = tgt_mask)
        
        # print('run mt_preds ...')
        mt_preds = tgt_decoder(contextual_feature = textual_contextual_feature, input = image, text = tgt_text[:, :-1], tgt_mask = tgt_mask)
        mt_preds_teacher = tgt_decoder(contextual_feature = textual_contextual_feature_teacher, input = image, text = tgt_text[:, :-1], tgt_mask = tgt_mask)

        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        ## Textual Loading Part
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # ###### Textual Parallel side data forward ...
        if opt.external_mt == 'yes':
            external_textual_feature = textual_encoder(input = image, text = textual_src_text[:, :-1], tgt_mask = tgt_mask)
            external_textual_contextual_feature = encoder(external_textual_feature, input = image, text = textual_src_text[:, :-1], tgt_mask = tgt_mask)
            external_mt_preds = tgt_decoder(contextual_feature = external_textual_contextual_feature, input = image, text = textual_tgt_text[:, :-1], tgt_mask = tgt_mask)
        
        # Save ground truth results both for triple and parallel
        src_target = src_text[:, 1:]
        tgt_target = tgt_text[:, 1:]  # without [GO] Symbol
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        ## Textual Loading Part
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        if opt.external_mt == 'yes':
            textual_src_target = textual_src_text[:, 1:]
            textual_tgt_target = textual_tgt_text[:, 1:]
        
        # In Deep-text original code, using logit to calculate loss directly
        # cost calculation for triple
        timt_cost_1 = criterion(timt_preds_1.contiguous().view(-1, timt_preds_1.shape[-1]), tgt_target.contiguous().view(-1))
        timt_cost_2 = criterion(timt_preds_2.contiguous().view(-1, timt_preds_2.shape[-1]), tgt_target.contiguous().view(-1))
        timt_cost_3 = criterion(timt_preds_3.contiguous().view(-1, timt_preds_3.shape[-1]), tgt_target.contiguous().view(-1))

        timt_cost = (timt_cost_1 + timt_cost_2 + timt_cost_3) / 3.0

        # cost calculation for parallel
        if opt.external_mt == 'yes':
            external_mt_cost = criterion(external_mt_preds.contiguous().view(-1, external_mt_preds.shape[-1]), textual_tgt_target.contiguous().view(-1))
        
        mt_cost = criterion(mt_preds.contiguous().view(-1, mt_preds.shape[-1]), tgt_target.contiguous().view(-1))
        mt_cost_teacher = criterion(mt_preds_teacher.contiguous().view(-1, mt_preds_teacher.shape[-1]), tgt_target.contiguous().view(-1))
        mt_cost = (mt_cost + mt_cost_teacher) / 2.0

        if opt.CL_Task:
            fv_1 = torch.sum(visual_contextual_feature_1, dim = 1) / visual_contextual_feature_1.shape[1]
            fv_2 = torch.sum(visual_contextual_feature_2, dim = 1) / visual_contextual_feature_2.shape[1]
            fv_3 = torch.sum(visual_contextual_feature_3, dim = 1) / visual_contextual_feature_3.shape[1]
            ft = torch.sum(textual_contextual_feature, dim = 1) / textual_contextual_feature.shape[1]
            ft_teacher = torch.sum(textual_contextual_feature_teacher, dim = 1) / textual_contextual_feature_teacher.shape[1]

            if opt.CL_type == 'mcl':
                cl_cost_vv_1 = contrastive_loss_cal_mcl(fv_1, fv_2, opt.CL_tau_vv)
                cl_cost_vv_2 = contrastive_loss_cal_mcl(fv_2, fv_3, opt.CL_tau_vv)
                cl_cost_vv_3 = contrastive_loss_cal_mcl(fv_3, fv_1, opt.CL_tau_vv)
                cl_cost_vv = (cl_cost_vv_1 + cl_cost_vv_2 + cl_cost_vv_3 ) / 3.0

                cl_cost_vl_1 = contrastive_loss_cal_mcl(fv_1, ft, opt.CL_tau_vl)
                cl_cost_vl_2 = contrastive_loss_cal_mcl(fv_2, ft, opt.CL_tau_vl)
                cl_cost_vl_3 = contrastive_loss_cal_mcl(fv_3, ft, opt.CL_tau_vl)
                cl_cost_vl = (cl_cost_vl_1 + cl_cost_vl_2 + cl_cost_vl_3 ) / 3.0

                cl_cost_ll = contrastive_loss_cal_mcl(ft, ft_teacher, opt.CL_tau_ll)

                cl_cost = opt.CL_vv_Weight * cl_cost_vv + opt.CL_vl_Weight * cl_cost_vl + opt.CL_ll_Weight * cl_cost_ll

            else:
                print('No clear contrastive loss type defined. Set cl_cost to 0!!!')
                cl_cost = 0
            
        else:
            AssertionError("Please set CL_Task store_true in this code version")
        
        if opt.CL_Task:
            if opt.MT_Weight > -1:
                MT_Weight = opt.MT_Weight
                CL_Weight = opt.CL_Weight
                TIMT_Weight = 1 - MT_Weight - CL_Weight
            else:
                print('Please input correct loss weights of machine translation task.')
                exit()
            
            if opt.MT_Weight >= 1.0:
                MT_Weight = opt.MT_Weight
                CL_Weight = opt.CL_Weight
                TIMT_Weight = 1
            
            weighted_timt_cost = TIMT_Weight * timt_cost
            weighted_mt_cost = MT_Weight * mt_cost
            weighted_cl_cost = CL_Weight * cl_cost
            if opt.external_mt == 'yes':
                cost = weighted_timt_cost + weighted_mt_cost + weighted_cl_cost + external_mt_cost
            else:
                cost = weighted_timt_cost + weighted_mt_cost + weighted_cl_cost
        else:
            AssertionError("Please set CL_Task store_true in this code version")

        visual_encoder.zero_grad()
        textual_encoder.zero_grad()
        encoder.zero_grad()
        tgt_decoder.zero_grad()

        cost.backward()
        torch.nn.utils.clip_grad_norm_(visual_encoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(textual_encoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(tgt_decoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)

        optimizer.step()

        loss_avg.add(weighted_timt_cost)
        loss_avg.add(weighted_mt_cost)
        loss_avg.add(weighted_cl_cost)
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        ## Textual Loading Part
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        if opt.external_mt == 'yes':
            loss_avg.add(external_mt_cost)
        timt_loss_avg.add(weighted_timt_cost)
        mt_loss_avg.add(weighted_mt_cost)
        cl_loss_avg.add(weighted_cl_cost)
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        ## Textual Loading Part
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        if opt.external_mt == 'yes':
            external_mt_loss_avg.add(external_mt_cost)
        
        # print loss at each step ...
        duration_time = time.time() - old_time
        if opt.external_mt == 'yes':
            print_str=f'step = {iteration+1}, loss = {loss_avg.val():0.5f}, timt_loss = {timt_loss_avg.val():0.5f}, cl_loss = {cl_loss_avg.val():0.5f}, mt_loss = {mt_loss_avg.val():0.5f}, external_mt_loss: {external_mt_loss_avg.val():0.5f}, duration = {duration_time:0.2f}s'
        else:
            print_str=f'step = {iteration+1}, loss = {loss_avg.val():0.5f}, timt_loss = {timt_loss_avg.val():0.5f}, cl_loss = {cl_loss_avg.val():0.5f}, mt_loss = {mt_loss_avg.val():0.5f}, duration = {duration_time:0.2f}s'
        
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        ## Textual Loading Part
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        old_time = time.time()
        print(print_str)
        print('-' * 100)
        
        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            print('-' * 80)
            print('Now in validation on iteration {} ...'.format(iteration + 1))
            elapsed_time = time.time() - start_time
                
            visual_encoder.eval()
            textual_encoder.eval()
            encoder.eval()
            tgt_decoder.eval()
            model_list = [visual_encoder, textual_encoder, encoder, tgt_decoder]

            with torch.no_grad():
                valid_loss, current_accuracy, current_bleu, timt_preds_str, mt_preds_str, src_labels, tgt_labels, infer_time, length_of_data = validation_modal_contrastive_learning_timt_task(
                    model_list, criterion, valid_loader, src_converter, tgt_converter, opt)
            
            visual_encoder.train()
            textual_encoder.train()
            encoder.train()
            tgt_decoder.train()

            loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
            loss_avg.reset()

            current_model_log = f'{"Current_valid_loss":17s}: {valid_loss:0.5f}, {"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_bleu":17s}: {current_bleu:0.3f}'

            # keep best accuracy model (on valid dataset)
            if valid_loss <= best_valid_loss:
                print('Saving best_valid_loss model ...')
                best_valid_loss = valid_loss
                torch.save(visual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'visual_encoder' + '.pth')
                torch.save(textual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'textual_encoder' + '.pth')
                torch.save(encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'encoder' + '.pth')
                torch.save(tgt_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'tgt_decoder' + '.pth')
                
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'visual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'visual_encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'textual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'textual_encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'tgt_decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'tgt_decoder' + '.pth')

                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'visual_encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'textual_encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'tgt_decoder' + '.pth')

                previous_best_valid_iter = iteration + 1
            
            # keep best accuracy model (on valid dataset)  
            if current_accuracy >= best_accuracy:
                print('Saving best_accuracy model ...')
                best_accuracy = current_accuracy
                torch.save(visual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'visual_encoder' + '.pth')
                torch.save(textual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'textual_encoder' + '.pth')
                torch.save(encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'encoder' + '.pth')
                torch.save(tgt_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'tgt_decoder' + '.pth')

                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'visual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'visual_encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'textual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'textual_encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'tgt_decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'tgt_decoder' + '.pth')

                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'visual_encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'textual_encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'tgt_decoder' + '.pth')
                
                previous_best_accuracy_iter = iteration + 1
            
            # keep best bleu model (on valid dataset)  
            print('Current bleu: {}'.format(current_bleu))
            print('Current best_bleu: {}'.format(best_bleu))
            if current_bleu >= best_bleu:
                print('Saving best_bleu model ...')
                best_bleu = current_bleu
                torch.save(visual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'visual_encoder' + '.pth')
                torch.save(textual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'textual_encoder' + '.pth')
                torch.save(encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'encoder' + '.pth')
                torch.save(tgt_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'tgt_decoder' + '.pth')
                
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'visual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'visual_encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'textual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'textual_encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'encoder' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'tgt_decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'tgt_decoder' + '.pth')

                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'visual_encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'textual_encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'encoder' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'tgt_decoder' + '.pth')
                
                previous_best_bleu_iter = iteration + 1
            
            best_model_log = f'{"Best_valid":17s}: {best_valid_loss:0.5f}, {"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_blue":17s}: {best_bleu:0.2f}'

            loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
            print(loss_model_log)

        # save model per opt.saveInterval
        if (iteration + 1) % opt.saveInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            print('-' * 80)
            print('Saving model on Step of {} ...'.format(iteration + 1))

            torch.save(visual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'visual_encoder' + '.pth')
            torch.save(textual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'textual_encoder' + '.pth')
            torch.save(encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'encoder' + '.pth')
            torch.save(tgt_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'tgt_decoder' + '.pth')
        
        # Final Step and offer information
        if (iteration + 1) == opt.num_iter:
            print('end the training at step {}!'.format(iteration + 1))

            print('Remove iter_step_1_* model savings, which is just a model saving to see whether it could run normally.')
            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'visual_encoder' + '.pth')
            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'textual_encoder' + '.pth')
            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'encoder' + '.pth')
            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'tgt_decoder' + '.pth')
            
            sys.exit()


import time
import torch
import os
import math
import torch.nn as nn
import numpy as np
from models.AGM.AGM_task import AGM_task, AGM_Config
from sklearn.metrics import accuracy_score
from models.AGM.function_tools import save_config, get_device, get_logger, set_seed, weight_init
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn import linear_model

def Accuracy(logits, target):
    logits = logits.detach().cpu()
    target = target.detach().cpu()

    preds = logits.argmax(dim=-1)
    assert preds.shape == target.shape
    correct = torch.sum(preds==target)
    total = torch.numel(target)
    if total == 0:
        return 1
    else:
        return correct / total


def train_epoch(model, train_dataloader, optimizer, scheduler, logger, cfgs, epoch, device, writer, last_score_a,
          last_score_v, audio_lr_ratio, visual_lr_ratio):
    # print(f'1 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    model.train()
    total_batch = len(train_dataloader)
    n_classes = 31

    model.mode = 'train'
    train_score_a = last_score_a
    train_score_v = last_score_v
    ra_score_a = 0.
    ra_score_v = 0.
    train_batch_loss = 0.

    num = [0.0 for _ in range(n_classes)]
    acc = [0.0 for _ in range(n_classes)]
    acc_a = [0.0 for _ in range(n_classes)]
    acc_v = [0.0 for _ in range(n_classes)]
    _loss = 0.
    for step, (image, spec, label) in enumerate(train_dataloader):
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        iteration = (epoch - 1) * total_batch + step + 1

        out, out_a, out_v, out_pad, out_ = model(spec.unsqueeze(1).float(), image.float())
        # print(f'2 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')
        loss = criterion(out, label)
        _loss += loss.item()

        train_batch_loss += loss.item() / total_batch

        writer.add_scalar('loss/step', loss, (epoch - 1) * total_batch + step)

        # calculate acc
        prediction = softmax(out)
        pred_a = softmax(out_a)
        pred_v = softmax(out_v)
        for j in range(image.shape[0]):
            ma = np.argmax(prediction[j].cpu().data.numpy())
            v = np.argmax(pred_v[j].cpu().data.numpy())
            a = np.argmax(pred_a[j].cpu().data.numpy())
            num[label[j]] += 1.0

            if np.asarray(label[j].cpu()) == ma:
                acc[label[j]] += 1.0
            if np.asarray(label[j].cpu()) == v:
                acc_v[label[j]] += 1.0
            if np.asarray(label[j].cpu()) == a:
                acc_a[label[j]] += 1.0

        optimizer.zero_grad()
        if torch.isnan(out_a).any() or torch.isnan(out_v).any():
            raise ValueError
        score_audio = 0.
        score_visual = 0.
        for k in range(out_a.size(0)):
            if torch.isinf(torch.log(softmax(out_a)[k][label[k]])) or softmax(out_a)[k][label[k]] < 1e-8:
                score_audio += - torch.log(torch.tensor(1e-8, dtype=out_a.dtype, device=out_a.device))
            else:
                score_audio += - torch.log(softmax(out_a)[k][label[k]])

            if torch.isinf(torch.log(softmax(out_v)[k][label[k]])) or softmax(out_v)[k][label[k]] < 1e-8:
                score_visual += - torch.log(torch.tensor(1e-8, dtype=out_v.dtype, device=out_v.device))
            else:
                score_visual += - torch.log(softmax(out_v)[k][label[k]])
        score_audio = score_audio / out_a.size(0)
        score_visual = score_visual / out_v.size(0)

        ratio_a = math.exp(score_visual.item() - score_audio.item())
        ratio_v = math.exp(score_audio.item() - score_visual.item())

        optimal_ratio_a = math.exp(train_score_v - train_score_a)
        optimal_ratio_v = math.exp(train_score_a - train_score_v)

        coeff_a = math.exp(cfgs.alpha * (min(optimal_ratio_a - ratio_a, 10)))
        coeff_v = math.exp(cfgs.alpha * (min(optimal_ratio_v - ratio_v, 10)))

        train_score_a = train_score_a * (iteration - 1) / iteration + score_audio.item() / iteration
        train_score_v = train_score_v * (iteration - 1) / iteration + score_visual.item() / iteration
        ra_score_a = ra_score_a * step / (step + 1) + score_audio.item() / (step + 1)
        ra_score_v = ra_score_v * step / (step + 1) + score_visual.item() / (step + 1)

        if cfgs.method == "AGM" and cfgs.modulation_starts <= epoch <= cfgs.modulation_ends:
            if step == 0:
                logger.info('Using AGM methods...')
            else:
                model.update_scale(coeff_a, coeff_v)
            loss.backward()
            # print(f'3 cuda allocated: {torch.cuda.memory_allocated() // (1024 ** 3)} GB')
        else:
            # normal
            if step == 0:
                logger.info('Using normal methods...')
            loss.backward()

        # if cfgs.method == "AGM" or cfgs.fusion_type == "early_fusion":
        #     if cfgs.fusion_type == "late_fusion":
        #         grad_max = torch.max(model.net.fusion_module.fc_out.weight.grad)
        #         grad_min = torch.min(model.net.fusion_module.fc_out.weight.grad)
        #         if grad_max > 1 or grad_min < -1:
        #             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #     else:
        #         grad_max = torch.max(model.net.head.fc.weight.grad)
        #         grad_min = torch.min(model.net.head.fc.weight.grad)
        #         if grad_max > 1 or grad_min < -1:
        #             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 100 == 0:
            logger.info(
                'EPOCH:[{:3d}/{:3d}]--STEP:[{:5d}/{:5d}]--{}--Loss:{:.4f}--lr:{}'.format(epoch, cfgs.epochs, step,
                                                                                         total_batch, 'Train',
                                                                                         loss.item(),
                                                                                         [group['lr'] for group in
                                                                                          optimizer.param_groups]))

        # torch.cuda.empty_cache()
    scheduler.step()

    accuracy = sum(acc) / sum(num)
    accuracy_a = sum(acc_a) / sum(num)
    accuracy_v = sum(acc_v) / sum(num)

    writer.add_scalars('Epoch Accuracy(train)', {'accuracy': accuracy,
                                                 'accuracy audio': accuracy_a,
                                                 'accuracy visual': accuracy_v}, epoch)
    logger.info(
        'EPOCH:[{:3d}/{:3d}]--{}--acc:{:.4f}--acc_a:{:.4f}--acc_v:{:.4f}-Alpha:{}'.format(epoch, cfgs.epochs,
                                                                                          'Train', accuracy,
                                                                                          accuracy_a, accuracy_v,
                                                                                          cfgs.alpha))
    return _loss/len(train_dataloader), train_score_a, train_score_v


def test(model, test_dataloader, logger, cfgs, epoch, device, writer):
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    n_classes = 31
    start_time = time.time()

    with torch.no_grad():
        model.eval()
        model.mode = 'eval'
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        valid_score_a = 0.
        valid_score_v = 0.
        test_loss = 0.
        test_audio_loss = 0.
        test_visual_loss = 0.

        total_batch = len(test_dataloader)

        for step, (image, spec, label) in enumerate(test_dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            out, out_a, out_v, out_pad, out_ = model(spec.unsqueeze(1).float(), image.float())
            loss = criterion(out, label)
            loss_a = criterion(out_a, label)
            loss_v = criterion(out_v, label)

            score_audio = 0.
            score_visual = 0.
            for k in range(out_a.size(0)):
                if torch.isinf(torch.log(softmax(out_a)[k][label[k]])) or softmax(out_a)[k][label[k]] < 1e-8:
                    score_audio += - torch.log(torch.tensor(1e-8, dtype=out_a.dtype, device=out_a.device))
                else:
                    score_audio += - torch.log(softmax(out_a)[k][label[k]])
                if torch.isinf(torch.log(softmax(out_v)[k][label[k]])) or softmax(out_v)[k][label[k]] < 1e-8:
                    score_visual += - torch.log(torch.tensor(1e-8, dtype=out_v.dtype, device=out_v.device))
                else:
                    score_visual += - torch.log(softmax(out_v)[k][label[k]])
            score_audio = score_audio / out_a.size(0)
            score_visual = score_visual / out_v.size(0)

            valid_score_a = valid_score_a * step / (step + 1) + score_audio.item() / (step + 1)
            valid_score_v = valid_score_v * step / (step + 1) + score_visual.item() / (step + 1)

            ratio_a = math.exp(valid_score_v - valid_score_a)
            ratio_v = math.exp(valid_score_a - valid_score_v)

            test_loss += loss.item() / total_batch
            test_audio_loss += loss_a.item() / total_batch
            test_visual_loss += loss_v.item() / total_batch

            iteration = (epoch - 1) * total_batch + step
            writer.add_scalar('test (loss/step)', loss, iteration)


            prediction = softmax(out)
            pred_a = softmax(out_a)
            pred_v = softmax(out_v)

            for j in range(image.shape[0]):
                ma = np.argmax(prediction[j].cpu().data.numpy())
                v = np.argmax(pred_v[j].cpu().data.numpy())
                a = np.argmax(pred_a[j].cpu().data.numpy())
                num[label[j]] += 1.0

                if np.asarray(label[j].cpu()) == ma:
                    acc[label[j]] += 1.0
                if np.asarray(label[j].cpu()) == v:
                    acc_v[label[j]] += 1.0
                    acc_a[label[j]] += 1.0

            if step % 20 == 0:
                logger.info('EPOCH:[{:03d}/{:03d}]--STEP:[{:05d}/{:05d}]--{}--loss:{}'.format(epoch, cfgs.epochs, step,
                                                                                              total_batch, 'Validate',
                                                                                              loss))


    accuracy = sum(acc) / sum(num)
    accuracy_a = sum(acc_a) / sum(num)
    accuracy_v = sum(acc_v) / sum(num)
    writer.add_scalars('Accuracy(Test)', {'accuracy': accuracy,
                                          'audio_accuracy': accuracy_a,
                                          'visual accuracy': accuracy_v}, epoch)

    model.mode = 'train'
    end_time = time.time()
    elapse_time = end_time - start_time
    logger.info(
        'EPOCH:[{:03d}/{:03d}]--{}--Elapse time:{:.2f}--Accuracy:{:.4f}--acc_a:{:.4f}--acc_v:{:.4f}'.format(epoch,
                                                                                                            cfgs.epochs,
                                                                                                            'Validate',
                                                                                                            elapse_time,
                                                                                                            accuracy,
                                                                                                            accuracy_a,
                                                                                                            accuracy_v))
    return accuracy, accuracy_a, accuracy_v, test_loss, test_audio_loss, test_visual_loss


def test_compute_weight(model, test_dataloader, logger, cfgs, epoch, device, writer, mm_to_audio_lr, mm_to_visual_lr,
                        test_audio_out, test_visual_out):
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    n_classes = 31
    start_time = time.time()
    test_loss = 0.
    test_audio_loss = 0.
    test_visual_loss = 0.
    with torch.no_grad():
        model.eval()
        model.mode = 'eval'

        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]
        valid_score_a = 0.
        valid_score_v = 0.
        total_batch = len(test_dataloader)
        model.extract_mm_feature = True
        ota = []
        otv = []

        for step, (image, spec, label) in enumerate(test_dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            out_a, out_v, out, encoded_feature = model(spec.unsqueeze(1).float(), image.float())
            out_to_audio = mm_to_audio_lr.predict(encoded_feature.detach().cpu())
            out_to_visual = mm_to_visual_lr.predict(encoded_feature.detach().cpu())
            ota.append(torch.from_numpy(out_to_audio))
            otv.append(torch.from_numpy(out_to_visual))
            loss = criterion(out, label)
            loss_a = criterion(out_a, label)
            loss_v = criterion(out_v, label)

            score_audio = 0.
            score_visual = 0.
            for k in range(out_a.size(0)):
                if torch.isinf(torch.log(softmax(out_a)[k][label[k]])) or softmax(out_a)[k][label[k]] < 1e-8:
                    score_audio += - torch.log(torch.tensor(1e-8, dtype=out_a.dtype, device=out_a.device))
                else:
                    score_audio += - torch.log(softmax(out_a)[k][label[k]])
                if torch.isinf(torch.log(softmax(out_v)[k][label[k]])) or softmax(out_v)[k][label[k]] < 1e-8:
                    score_visual += - torch.log(torch.tensor(1e-8, dtype=out_v.dtype, device=out_v.device))
                else:
                    score_visual += - torch.log(softmax(out_v)[k][label[k]])
            score_audio = score_audio / out_a.size(0)
            score_visual = score_visual / out_v.size(0)

            valid_score_a = valid_score_a * step / (step + 1) + score_audio.item() / (step + 1)
            valid_score_v = valid_score_v * step / (step + 1) + score_visual.item() / (step + 1)

            ratio_a = math.exp(valid_score_v - valid_score_a)
            ratio_v = math.exp(valid_score_a - valid_score_v)

            test_loss += loss.item() / total_batch
            test_audio_loss += loss_a.item() / total_batch
            test_visual_loss += loss_v.item() / total_batch

            iteration = (epoch - 1) * total_batch + step
            writer.add_scalar('test (loss/step)', loss, iteration)

            prediction = softmax(out)
            pred_a = softmax(out_a)
            pred_v = softmax(out_v)

            for j in range(image.shape[0]):
                ma = np.argmax(prediction[j].cpu().data.numpy())
                v = np.argmax(pred_v[j].cpu().data.numpy())
                a = np.argmax(pred_a[j].cpu().data.numpy())
                num[label[j]] += 1.0

                if np.asarray(label[j].cpu()) == ma:
                    acc[label[j]] += 1.0
                if np.asarray(label[j].cpu()) == v:
                    acc_v[label[j]] += 1.0
                if np.asarray(label[j].cpu()) == a:
                    acc_a[label[j]] += 1.0

            if step % 20 == 0:
                logger.info('EPOCH:[{:03d}/{:03d}]--STEP:[{:05d}/{:05d}]--{}--loss:{}'.format(epoch, cfgs.epochs, step,
                                                                                              total_batch, 'Validate',
                                                                                              loss))
        model.extract_mm_feature = False
        # ota = torch.cat(ota, dim=0).float()
        # otv = torch.cat(otv, dim=0).float()
        # ota = ota - test_audio_out
        # otv = otv - test_visual_out
        #
        # ba = torch.cov(test_audio_out.T) * test_audio_out.size(0)
        # bv = torch.cov(test_visual_out.T) * test_visual_out.size(0)

        # ra = torch.trace((ota @ torch.pinverse(ba)).T @ ota) / test_audio_out.size(1)
        # rv = torch.trace((otv @ torch.pinverse(bv)).T @ otv) / test_visual_out.size(1)
        accuracy = sum(acc) / sum(num)
        accuracy_a = sum(acc_a) / sum(num)
        accuracy_v = sum(acc_v) / sum(num)
        writer.add_scalars('Accuracy(Test)', {'accuracy': accuracy,
                                              'audio_accuracy': accuracy_a,
                                              'visual accuracy': accuracy_v}, epoch)

        model.mode = 'train'
        end_time = time.time()
        elapse_time = end_time - start_time
        logger.info(
            'EPOCH:[{:03d}/{:03d}]--{}--Elapse time:{:.2f}--Accuracy:{:.4f}--acc_a:{:.4f}--acc_v:{:.4f}'.format(epoch,
                                                                                                                cfgs.epochs,
                                                                                                                'Validate',
                                                                                                                elapse_time,
                                                                                                                accuracy,
                                                                                                                accuracy_a,
                                                                                                                accuracy_v))
        return accuracy, accuracy_a, accuracy_v, test_audio_loss, test_visual_loss

def extract_mm_feature(model, dep_dataloader, device, cfgs):
    all_feature = []
    model.eval()
    model.mode = 'eval'
    with torch.no_grad():
        total_batch = len(dep_dataloader)
        for step, (image, spec, label) in enumerate(dep_dataloader):
            spec = spec.to(device)
            image = image.to(device)
            # label = label.to(device)
            model.extract_mm_feature = True
            out_a, out_v, out, feature = model(spec.unsqueeze(1).float(),image.float())
            all_feature.append(feature)
            model.extract_mm_feature = False
        all_feature = torch.cat(all_feature, dim=0)
        return all_feature


def AGM_main(args):
    cfgs = AGM_Config(args)
    device = torch.device('cuda:0')
    gpu_ids = list(range(torch.cuda.device_count()))
    ts = time.strftime('%Y_%m_%d %H:%M:%S', time.localtime())
    print(ts)
    save_dir = os.path.join(cfgs.ckpt_path, f"{ts}_{cfgs.method}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = get_logger("train_logger", logger_dir=save_dir)
    save_config(cfgs, save_dir)

    task = AGM_task(cfgs)
    train_dataloader = task.train_dataloader
    val_dataloader = task.valid_dataloader
    test_dataloader = task.test_dataloader
    
    model = task.model
    optimizer = task.optimizer
    scheduler = task.scheduler
    # model.to(device)
    model.cuda(device)

    # if cfgs.use_mgpu:
    #     model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    #     model.cuda()

    mm = []
    a = []
    v = []

    cos_a_all = []
    cos_v_all = []

    best_epoch = {'epoch':0,'acc':0.,'train_loss':0.,'valid_loss':0.}
    start_epoch = 1
    # if cfgs.breakpoint_path is not None:
    #     ckpt = torch.load(cfgs.breakpoint_path)
    #     model.load_state_dict(ckpt['state_dict'])
    #     optimizer.load_state_dict(ckpt['optimizer'])
    #     best_epoch['epoch'] = ckpt['epoch']
    #     best_epoch['acc'] = ckpt['acc']
    #     start_epoch = ckpt['epoch'] + 1
    epoch_score_a = 0.
    epoch_score_v = 0.
    best_epoch = {'epoch': 0, 'acc': 0.}

    audio_lr_ratio = 1.0
    visual_lr_ratio = 1.0
    audio_lr_memory = []
    visual_lr_memory = []
    min_test_audio_loss = 1000
    min_test_visual_loss = 1000
    audio_argmax_loss_epoch = 0.
    visual_argmax_loss_epoch = 0.
    best_acc = 0.0001 # todo choose a reasonable value
    print('init best acc: ', best_acc)

    writer_path = os.path.join(cfgs.tensorboard_path)
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)
    log_name = 'model_{}_{}_{}_epoch{}_batch{}_lr{}_alpha{}'.format(
        cfgs.method, cfgs.optimizer, cfgs.dataset, cfgs.epochs, cfgs.batch_size, cfgs.learning_rate, cfgs.alpha)
    writer = SummaryWriter(os.path.join(writer_path, log_name))

    for epoch in range(1, cfgs.epochs):
        logger.info(f'Training for epoch {epoch}...')
        epoch_loss, epoch_score_a, epoch_score_v = train_epoch(model, train_dataloader, optimizer, scheduler, logger, cfgs, epoch, device, writer,
                                             epoch_score_a, epoch_score_v, audio_lr_ratio, visual_lr_ratio)

        # model, train_dataloader, optimizer, scheduler, logger, cfgs, epoch, device, writer, last_score_a,
        # last_score_v, audio_lr_ratio, visual_lr_ratio

        test_acc, accuracy_a, accuracy_v, test_loss, validate_audio_batch_loss,\
            validate_visual_batch_loss = test(model, val_dataloader, logger, cfgs, epoch, device, writer)

        if test_acc > best_acc:
            best_acc = float(test_acc)

            if not os.path.exists(args.ckpt_path):
                os.mkdir(args.ckpt_path)

            model_name = 'best_model_{}_of_{}_epoch{}_batch{}_lr{}_alpha{}.pth'.format(
                args.method, args.optimizer, args.epochs, args.batch_size, args.learning_rate, args.alpha)

            saved_dict = {'saved_epoch': epoch,
                          'acc': test_acc,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict()}

            save_dir = os.path.join(args.ckpt_path, model_name)
            torch.save(saved_dict, save_dir)
            logger.info('The best model has been saved at {}.'.format(save_dir))
        else:
            logger.info(
                "Loss: {:.4f}, Acc: {:.4f}, Acc_f: {:.4f}, Acc_v: {:.4f},Best Acc: {:.4f}, vloss: {:.4f}".format(
                    epoch_loss, test_acc, accuracy_a, accuracy_v, best_acc, test_loss))


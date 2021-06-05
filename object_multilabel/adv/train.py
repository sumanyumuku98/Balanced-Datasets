import math, os, random, json, pickle, sys, pdb
import string, shutil, time, argparse
import numpy as np
import itertools

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tqdm import tqdm as tqdm
from PIL import Image

import torch.nn.functional as F
import torch, torchvision
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
from torch.utils.data import DataLoader

from data_loader import CocoObjectGender
from model import ObjectMultiLabelAdv
from logger import Logger

object_id_map = pickle.load(open('../data/object_id.map'))
object2id = object_id_map['object2id']
id2object = object_id_map['id2object']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str,
            help='path for saving checkpoints')
    parser.add_argument('--log_dir', type=str,
            help='path for saving log files')

    parser.add_argument('--ratio', type=str,
            default = '0')
    parser.add_argument('--num_object', type=int,
            default = 79)

    parser.add_argument('--annotation_dir', type=str,
            default='../data',
            help='annotation files path')
    parser.add_argument('--image_dir',
            default = '../data',
            help='image directory')

    parser.add_argument('--balanced', action='store_true',
            help='use balanced subset for training, ratio will be 1/2/3')
    parser.add_argument('--batch_balanced', action='store_true',
            help='in every batch, gender balanced')
    parser.add_argument('--gender_balanced', action='store_true',
            help='use gender balanced subset for training')

    parser.add_argument('--adv_on', action='store_true',
            help='start adv training')
    parser.add_argument('--layer', type=str,
            help='extract image feature for adv at this layer')
    parser.add_argument('--adv_conv', action='store_true',
            help='add conv layers to adv component')
    parser.add_argument('--no_avgpool', action='store_true',
            help='remove avgpool layer for adv component')
    parser.add_argument('--adv_capacity', type=int,
            help='linear layer dimension for adv component')
    parser.add_argument('--adv_lambda', type=float,
            help='weight assigned to adv loss')
    parser.add_argument('--dropout', type=float,
            help='parameter for dropout layter in adv component')

    parser.add_argument('--blackout', action='store_true')
    parser.add_argument('--blackout_box', action='store_true')
    parser.add_argument('--blackout_face', action='store_true')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--edges', action='store_true')

    parser.add_argument('--no_image', action='store_true',
            help='do not load image in dataloaders')

    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument("--use_fair", action="store_true")


    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.save_dir = os.path.join('./models', args.layer + '_' + str(args.adv_capacity) + '_' + \
            str(args.adv_lambda) + '_' + str(args.dropout) + '_' + args.save_dir)

    if os.path.exists(args.save_dir) and not args.resume:
        print('Path {} exists! and not resuming'.format(args.save_dir))
        return
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    args.log_dir = os.path.join('./logs', args.layer + '_' + str(args.adv_capacity) + '_' + \
            str(args.adv_lambda) + '_' + str(args.dropout) + '_' + args.log_dir)

    train_log_dir = os.path.join(args.log_dir, 'train')
    val_log_dir = os.path.join(args.log_dir, 'val')
    if not os.path.exists(train_log_dir): os.makedirs(train_log_dir)
    if not os.path.exists(val_log_dir): os.makedirs(val_log_dir)
    train_logger = Logger(train_log_dir)
    val_logger = Logger(val_log_dir)

    # Save all parameters for training
    with open(os.path.join(args.log_dir, "arguments.txt"), "a") as f:
        f.write(str(args)+'\n')

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    # Image preprocessing
    train_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    val_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])

    # Data samplers.
    train_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir, split = 'train', transform = train_transform)
    val_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir, split = 'val', transform = val_transform)

    args.gender_balanced = True
    val_data_gender_balanced = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
            image_dir = args.image_dir, split = 'val', transform = val_transform)
    args.gender_balanced = False

    # Data loaders / batch assemblers.

    if args.batch_balanced:
        train_batch_size = int(2.5 * args.batch_size)
    else:
        train_batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = train_batch_size,
            shuffle = True, num_workers = 6, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size,
            shuffle = False, num_workers = 4, pin_memory = True)

    val_loader_gender_balanced = torch.utils.data.DataLoader(val_data_gender_balanced, \
        batch_size = args.batch_size, shuffle = False, num_workers = 4, pin_memory = True)


    # Build the models
    model = ObjectMultiLabelAdv(args, args.num_object, args.adv_capacity, args.dropout, \
        args.adv_lambda).cuda()

    object_weights = torch.FloatTensor(train_data.getObjectWeights())
    criterion = nn.BCEWithLogitsLoss(weight=object_weights, reduction='elementwise_mean').cuda()

    # print model
    def trainable_params():
        for param in model.parameters():
            if param.requires_grad:
                yield param

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num_trainable_params:', num_trainable_params)
    optimizer = torch.optim.Adam(trainable_params(), args.learning_rate, weight_decay = 1e-5)

    best_performance = 0
    if args.resume:
        if os.path.isfile(os.path.join(args.save_dir, 'checkpoint.pth.tar')):
            print("=> loading checkpoint '{}'".format(args.save_dir))
            checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pth.tar'))
            args.start_epoch = checkpoint['epoch']
            best_performance = checkpoint['best_performance']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.save_dir))

    print('before training, evaluate the model')
    test_balanced(args, 0, model, criterion, val_loader_gender_balanced, None, logging=False)
    test(args, 0, model, criterion, val_loader, None, logging=False)

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        train(args, epoch, model, criterion, train_loader, optimizer, train_logger, logging=True)
        test_balanced(args, epoch, model, criterion, val_loader_gender_balanced, val_logger, logging=True)
        current_score = test(args, epoch, model, criterion, val_loader, val_logger, logging=True)
        is_best = current_score > best_performance
        best_performance = max(current_score, best_performance)
        model_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_performance': best_performance}
        save_checkpoint(args, model_state, is_best, os.path.join(args.save_dir, \
                'checkpoint.pth.tar'))

        # save the model from the last epoch
        if epoch == args.num_epochs:
            torch.save(model_state, os.path.join(args.save_dir, \
                'checkpoint_%d.pth.tar' % args.num_epochs))

def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save_dir, 'model_best.pth.tar'))



def train(args, epoch, model, criterion, train_loader, optimizer, train_logger,  logging=True):
    model.train()
    task_preds, adv_preds = [], []
    task_truth, adv_truth = [], []
    nTrain = len(train_loader.dataset) # number of images
    task_loss_logger = AverageMeter()
    adv_loss_logger = AverageMeter()

    t = tqdm(train_loader, desc = 'Train %d' % epoch)
    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # for debugging

        # Set mini-batch dataset
        if args.batch_balanced:
            man_idx = genders[:, 0].nonzero().squeeze()
            if len(man_idx.size()) == 0: man_idx = man_idx.unsqueeze(0)
            woman_idx = genders[:, 1].nonzero().squeeze()
            if len(woman_idx.size()) == 0: woman_idx = woman_idx.unsqueeze(0)
            selected_num = min(len(man_idx), len(woman_idx))
            if selected_num < args.batch_size / 2:
                continue
            else:
                selected_num = args.batch_size / 2
                selected_idx = torch.cat((man_idx[:selected_num], woman_idx[:selected_num]), 0)

            images = torch.index_select(images, 0, selected_idx)
            targets = torch.index_select(targets, 0, selected_idx)
            genders = torch.index_select(genders, 0, selected_idx)

        images = images.cuda()
        targets = targets.cuda()
        genders = genders.cuda()

        # Forward, Backward and Optimizer
        task_pred, adv_pred = model(images) # if conv5, adv_pred.size is (batch_size, 49, 2)

        task_loss = criterion(task_pred, targets)
        adv_loss = F.cross_entropy(adv_pred, genders.max(1, keepdim=False)[1], reduction='elementwise_mean')

        task_loss_logger.update(task_loss.item())
        adv_loss_logger.update(adv_loss.item())

        adv_pred = np.argmax(F.softmax(adv_pred, dim=1).cpu().detach().numpy(), axis=1)
        adv_preds += adv_pred.tolist()
        adv_truth += genders.cpu().max(1, keepdim=False)[1].numpy().tolist()

        task_pred = torch.sigmoid(task_pred)

        if batch_idx > 0 and len(task_preds) > 0:
            task_preds = torch.cat((task_preds, task_pred.detach().cpu()), 0)
            task_truth = torch.cat((task_truth, targets.cpu()), 0)
            total_genders = torch.cat((total_genders, genders.cpu()), 0)
        else:
            task_preds = task_pred.detach().cpu()
            task_truth = targets.cpu()
            total_genders = genders.cpu()

        loss = task_loss + adv_loss

        # backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        t.set_postfix(loss = loss.item())

    task_f1_score = f1_score(task_truth.numpy(), (task_preds >= 0.5).long().numpy(), average = 'macro')
    meanAP = average_precision_score(task_truth.numpy(), task_preds.numpy(), average='macro')
    adv_acc = accuracy_score(adv_truth, adv_preds)

    if logging:
        train_logger.scalar_summary('task loss', task_loss_logger.avg, epoch)
        train_logger.scalar_summary('adv loss', adv_loss_logger.avg, epoch)
        train_logger.scalar_summary('task_f1_score', task_f1_score, epoch)
        train_logger.scalar_summary('meanAP', meanAP, epoch)
        train_logger.scalar_summary('adv acc', adv_acc, epoch)

    print('Train epoch  : {}, meanAP: {:.2f}, adv acc: {:.2f}, '.format(epoch, meanAP*100, adv_acc*100))


def test(args, epoch, model, criterion, val_loader, val_logger, logging=True):

    model.eval()
    task_preds, adv_preds = [], []
    task_truth, adv_truth = [], []
    nTest = len(val_loader.dataset) # number of images
    task_loss_logger = AverageMeter()
    adv_loss_logger = AverageMeter()

    t = tqdm(val_loader, desc = 'Val %d' % epoch)

    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # for debugging

        # Set mini-batch dataset
        images = images.cuda()
        targets = targets.cuda()
        genders = genders.cuda()

        task_pred, adv_pred = model(images)

        task_loss = criterion(task_pred, targets)
        adv_loss = F.cross_entropy(adv_pred, genders.max(1, keepdim=False)[1], reduction='elementwise_mean')

        task_loss_logger.update(task_loss.item())
        adv_loss_logger.update(adv_loss.item())

        adv_pred = np.argmax(F.softmax(adv_pred, dim=1).cpu().detach().numpy(), axis=1)
        adv_preds += adv_pred.tolist()
        adv_truth += genders.cpu().max(1, keepdim=False)[1].numpy().tolist()

        task_pred = torch.sigmoid(task_pred)

        if batch_idx > 0 and len(task_preds) > 0:
            task_preds = torch.cat((task_preds, task_pred.detach().cpu()), 0)
            task_truth = torch.cat((task_truth, targets.cpu()), 0)
            total_genders = torch.cat((total_genders, genders.cpu()), 0)
        else:
            task_preds = task_pred.detach().cpu()
            task_truth = targets.cpu()
            total_genders = genders.cpu()

        loss = task_loss + adv_loss

        # Print log info
        t.set_postfix(loss = loss.item())

    task_f1_score = f1_score(task_truth.numpy(), (task_preds >= 0.5).long().numpy(), average = 'macro')
    meanAP = average_precision_score(task_truth.numpy(), task_preds.numpy(), average='macro')
    adv_acc = accuracy_score(adv_truth, adv_preds)

    if logging:
        val_logger.scalar_summary('task loss', task_loss_logger.avg, epoch)
        val_logger.scalar_summary('adv loss', adv_loss_logger.avg, epoch)
        val_logger.scalar_summary('task_f1_score', task_f1_score, epoch)
        val_logger.scalar_summary('meanAP', meanAP, epoch)
        val_logger.scalar_summary('adv acc', adv_acc, epoch)

    print('Test epoch   : {}, meanAP: {:.2f}, adv acc: {:.2f}, '.format(epoch, meanAP*100, adv_acc*100))

    return task_f1_score

def test_balanced(args, epoch, model, criterion, val_loader, val_logger, logging=True):

    model.eval()
    task_preds, adv_preds = [], []
    task_truth, adv_truth = [], []
    nTest = len(val_loader.dataset) # number of images
    task_loss_logger = AverageMeter()
    adv_loss_logger = AverageMeter()

    t = tqdm(val_loader, desc = 'ValB %d' % epoch)

    for batch_idx, (images, targets, genders, image_ids) in enumerate(t):
        #if batch_idx == 100: break # for debugging

        # Set mini-batch dataset
        images = images.cuda()
        targets = targets.cuda()
        genders = genders.cuda()

        task_pred, adv_pred = model(images)

        task_loss = criterion(task_pred, targets)
        adv_loss = F.cross_entropy(adv_pred, genders.max(1, keepdim=False)[1], reduction='elementwise_mean')

        task_loss_logger.update(task_loss.item())
        adv_loss_logger.update(adv_loss.item())

        adv_pred = np.argmax(F.softmax(adv_pred, dim=1).cpu().detach().numpy(), axis=1)
        adv_preds += adv_pred.tolist()
        adv_truth += genders.cpu().max(1, keepdim=False)[1].numpy().tolist()

        task_pred = torch.sigmoid(task_pred)

        if batch_idx > 0 and len(task_preds) > 0:
            task_preds = torch.cat((task_preds, task_pred.detach().cpu()), 0)
            task_truth = torch.cat((task_truth, targets.cpu()), 0)
            total_genders = torch.cat((total_genders, genders.cpu()), 0)
        else:
            task_preds = task_pred.detach().cpu()
            task_truth = targets.cpu()
            total_genders = genders.cpu()

        loss = task_loss + adv_loss

        # Print log info
        t.set_postfix(loss = loss.item())

    task_f1_score = f1_score(task_truth.numpy(), (task_preds >= 0.5).long().numpy(), average = 'macro')
    meanAP = average_precision_score(task_truth.numpy(), task_preds.numpy(), average='macro')
    adv_acc = accuracy_score(adv_truth, adv_preds)

    if logging:
        val_logger.scalar_summary('adv loss balanced', adv_loss_logger.avg, epoch)
        val_logger.scalar_summary('adv acc balanced', adv_acc, epoch)

    print('Test epoch B : {}, meanAP: {:.2f}, adv acc: {:.2f}, '.format(epoch, meanAP*100, adv_acc*100))

    return task_f1_score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from fair_utils import ir_numpy
import math, os, random, json, pickle, sys, pdb
import string, shutil, time, argparse, operator, collections
import numpy as np
import logging
import argparse
from PIL import Image
import functools

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm as tqdm

from data_loader import CocoObjectGender
from model import GenderClassifier

object_id_map = pickle.load(open('./data/object_id.map'))
object2id = object_id_map['object2id']
id2object = object_id_map['id2object']

# print(object2id)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str,
            default='./dataset_leakage',
            help='path for saving checkpoints')

    parser.add_argument('--num_rounds', type=int,
            default = 5)

    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--ratio', type=str,
            default = '0')
    parser.add_argument('--num_object', type=int,
            default = 79)

    parser.add_argument('--annotation_dir', type=str,
            default='./data',
            help='annotation files path')
    parser.add_argument('--image_dir',
            default = './data',
            help='image directory')

    parser.add_argument('--hid_size', type=int,
            default = 300)

    parser.add_argument('--no_image', action='store_true')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    parser.add_argument('--print_every', type=int, default=500)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument("--gender_balanced", action="store_true")
    parser.add_argument("--use_fair", action="store_true")
    parser.add_argument("--name", type=str, default="placeholder")

    args = parser.parse_args()


    # print(args.balanced, args.use_fair)
    # args.gender_balanced = True # always True as we want to compute the leakage
    args.no_image = True

    args.blur = False
    args.blackout_face = False
    args.blackout = False
    args.blackout_box = False
    args.grayscale = False
    args.edges = False

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    test_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize])

    acc_list = []
    for i in range(args.num_rounds):

        train_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
                image_dir = args.image_dir,split = 'train', transform = train_transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
                    shuffle = True, num_workers = 6, pin_memory = True)

        print("Length of train dataset: %d" % len(train_data))
        # print("Man arr shape:", train_data.man_arr.shape)
        # print("Woman arr shape:", train_data.woman_arr.shape)
        ####
        # man_arr = train_data.man_arr
        # woman_arr=train_data.woman_arr
        # man_arr=np.sum(man_arr, axis=0)/man_arr.shape[0]
        # woman_arr=np.sum(woman_arr, axis=0)/woman_arr.shape[0]
        # man_ir=ir_numpy(man_arr)
        # woman_ir=ir_numpy(woman_arr)
        # print "Man imbalance ratio: %.3f" % man_ir
        # print "Woman imbalance ratio: %.3f" % woman_ir
        ####

        # Data samplersi for val set.
        val_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
                image_dir = args.image_dir,split = 'val', transform = test_transform)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size, \
                shuffle = False, num_workers = 4,pin_memory = True)

        print("Length of val dataset: %d" % len(val_data))

        # Data samplers for test set.
        test_data = CocoObjectGender(args, annotation_dir = args.annotation_dir, \
                image_dir = args.image_dir,split = 'test', transform = test_transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, \
                shuffle = False, num_workers = 4,pin_memory = True)

        print("Length of test dataset: %d" % len(test_data))

        # initialize gender classifier
        model = GenderClassifier(args, args.num_object)
        model = model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = 1e-5)

        model_save_dir = os.path.join(args.save_dir, args.name)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        train_genderclassifier(model, args.num_epochs, optimizer, train_loader, val_loader, \
            model_save_dir, args.print_every)

        model.load_state_dict(torch.load(model_save_dir+'/model_best.pth.tar')['state_dict'])
        loss, acc = epoch_pass(0, test_loader, model, None, False, print_every=500)
        loss, val_acc = epoch_pass(0, val_loader, model, None, False, print_every=500)
        acc = 0.5 + abs(acc - 0.5)
        val_acc = 0.5 + abs(val_acc - 0.5)
        print('round {} acc on test set: {}, val acc: {}'.format(i, acc*100, val_acc*100))
        acc_list.append(acc)
    
    logging.basicConfig(filename=os.path.join(model_save_dir, "leakage.log"), filemode='w', format='%(levelname)s:%(message)s', level=logging.INFO)

    print acc_list
    # acc_str=" "
    logging.info(acc_list)
    acc_ = np.array(acc_list)
    mean_acc = np.mean(acc_)
    std_acc = np.std(acc_)
    logging.info("Mean: %.3f and Std: %.3f" % (mean_acc, std_acc))
    print mean_acc, std_acc

def train_genderclassifier(model, num_epochs, optimizer, train_loader, test_loader, model_save_dir, \
    print_every):

    train_loss_arr = list()
    dev_loss_arr = list()
    train_acc_arr = list()
    val_acc_arr = list()

    best_score = 0

    for epoch in xrange(1, num_epochs + 1):

        # train
        train_loss, train_acc = epoch_pass(epoch, train_loader, model, optimizer, True, print_every)
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)
        if epoch % 10 == 0:
            print('train, {0}, train loss: {1:.2f}, train acc: {2:.2f}'.format(epoch, \
                train_loss*100, train_acc*100))

        # dev
        val_loss, val_acc = epoch_pass(epoch, test_loader, model, optimizer, False, print_every)
        dev_loss_arr.append(val_loss)
        val_acc_arr.append(val_acc)
        if epoch % 10 == 0:
            print('val, {0}, val loss: {1:.2f}, val acc: {2:.2f}'.format(epoch, val_loss*100, val_acc *100))

        if val_acc > best_score:
            best_score = val_acc
            best_model_epoch = epoch
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, \
                model_save_dir + '/model_best.pth.tar')
        if epoch % 10 == 0:
            print('current best dev score: {:.2f}'.format(best_score*100))


def epoch_pass(epoch, data_loader, model, optimizer, training, print_every=500):

    t_loss = 0.0
    n_processed = 0
    preds = list()
    truth = list()

    if training:
        model.train()
    else:
        model.eval()

    for ind, (_, targets, genders, image_ids) in enumerate(data_loader): # images are not provided

        targets = targets.cuda()
        genders = genders.cuda()

        predictions = model(targets)

        loss = F.cross_entropy(predictions, genders[:, 1], reduction='elementwise_mean')

        predictions = np.argmax(F.softmax(predictions, dim=1).cpu().detach().numpy(), axis=1)
        preds += predictions.tolist()
        truth += genders.cpu().max(1, keepdim=False)[1].numpy().tolist()

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_loss += loss.item()
        n_processed += len(genders)

        if (ind + 1) % print_every == 0:
            print('{0}: task loss: {1:4f}'.format(ind + 1, t_loss / n_processed))

    acc = accuracy_score(truth, preds)

    return t_loss / n_processed, acc

if __name__ == '__main__':
    main()



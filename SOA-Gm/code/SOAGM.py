from utils.utils import setup_seed,weight_init,re_init
from dataset.av_dataset import AVDataset_CD
import copy
from torch.utils.data import DataLoader
from models.models import AVClassifier
from sklearn import metrics
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import re
import numpy as np
from tqdm import tqdm
import argparse
import os
from scipy.optimize import minimize


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='KineticSound, CREMAD')
    parser.add_argument('--model', default='model', type=str)
    parser.add_argument('--n_classes', default=6, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=70, type=int)
    parser.add_argument('--optimizer', default='sgd',
                        type=str, choices=['sgd', 'adam','adamgrad'])
    parser.add_argument('--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=50, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1,
                        type=float, help='decay coefficient')
    parser.add_argument('--ckpt_path', default='log_cd',
                        type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--clip_grad', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--use_tensorboard', default=False,
                        type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='log_cd',
                        type=str, help='path to save tensorboard logs')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0,1,2,3',
                        type=str, help='GPU ids')

    return parser.parse_args()

def objective(weights, conf_a, conf_b, lambda_):
    a, b = weights
    social_welfare = a * conf_a + b * conf_b
    envy = max(0,a* conf_a - b* conf_b) + max(0,b* conf_b-a*conf_a)
    # envy = a * np.log(a/b) + b * np.log(b/a)
    return -social_welfare + lambda_ * envy
def constraint(weights):
    return 2 - sum(weights)

def calculate_belief(alpha, n_classes):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    b = E / S.expand(E.shape)
    u = n_classes / S
    return torch.mean(b,dim=0)


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Start training ... ")

    _loss = 0

    loss_value_mm = []
    loss_value_a = []
    loss_value_v = []

    record_names_audio = []
    record_names_visual = []
    for name, param in model.named_parameters():
        if 'head' in name:
            continue
        if ('audio' in name):
            record_names_audio.append((name, param))
            continue
        if ('visual' in name):
            record_names_visual.append((name, param))
            continue
    grads_history = {}

    for step, (spec, images, label) in tqdm(enumerate(dataloader)):

        optimizer.zero_grad()
        images = images.to(device)
        spec = spec.to(device)
        label = label.to(device)
        out, out_a, out_v = model(spec.float(), images.float())

        loss_mm = criterion(out, label)

        loss_a = criterion(out_a, label)

        loss_v = criterion(out_v, label)

        loss_value_mm.append(loss_mm.item())
        loss_value_a.append(loss_a.item())
        loss_value_v.append(loss_v.item())

        losses = [loss_mm, loss_a, loss_v]
        all_loss = ['both', 'audio', 'visual']

        grads_audio = {}
        grads_visual = {}

        for idx, loss_type in enumerate(all_loss):
            loss = losses[idx]
            loss.backward(retain_graph=True)

            if (loss_type == 'visual'):
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                grads_visual[loss_type]["concat"] = torch.cat(
                    [grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])

            elif (loss_type == 'audio'):
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio.keys():
                        grads_audio[loss_type] = {}
                    grads_audio[loss_type][tensor_name] = param.grad.data.clone()
                grads_audio[loss_type]["concat"] = torch.cat(
                    [grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])

            else:
                for tensor_name, param in record_names_audio:
                    if loss_type not in grads_audio.keys():
                        grads_audio[loss_type] = {}
                    grads_audio[loss_type][tensor_name] = param.grad.data.clone()
                grads_audio[loss_type]["concat"] = torch.cat(
                    [grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])
                for tensor_name, param in record_names_visual:
                    if loss_type not in grads_visual.keys():
                        grads_visual[loss_type] = {}
                    grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                grads_visual[loss_type]["concat"] = torch.cat(
                    [grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])

            optimizer.zero_grad()

        loss = loss_mm + loss_a + loss_v
        loss.backward()

        conf_out = torch.sum(torch.log(torch.sum(torch.exp(out),dim=1)),dim=0)
        conf_outa = torch.sum(torch.log(torch.sum(torch.exp(out_a),dim=1)),dim=0)
        conf_outv = torch.sum(torch.log(torch.sum(torch.exp(out_v),dim=1)),dim=0)
        bounds = [(0, 2), (0, 2)]
        bounds_v = [(0,2),(0,1.5)]
        initial_guess = [1, 1]
        constraints = {'type': 'ineq', 'fun': constraint}
        lambda_a = 0.7   # 示例值，可调整,sgd:0.7
        lambda_v = 0.65 #sgd:0.65
        result_a = minimize(objective, initial_guess, args=(conf_out.detach().item(), conf_outa.detach().item(), lambda_a),
                          bounds=bounds, constraints=constraints)
        if result_a.success:
            both_optimal_a, audio_optimal = result_a.x
            sw_a = both_optimal_a * conf_out + audio_optimal * conf_outa
            if both_optimal_a == 0:
                both_optimal_a = 1
            if audio_optimal == 0:
                audio_optimal = 1
            if sw_a < 0:
                sw_a = 128

            print(f"最优分配: a = {both_optimal_a:.4f}, b = {audio_optimal:.4f}")
            print(f"最大社会福利audio: {sw_a:.4f}")
        else:
            both_optimal_a = audio_optimal = 1
            sw_a = both_optimal_a * conf_out + audio_optimal * conf_outa
        result_v = minimize(objective, initial_guess, args=(conf_out.detach().item(), conf_outv.detach().item(), lambda_v),
                            bounds=bounds_v, constraints=constraints)
        if result_v.success:
            both_optimal_v, visual_optimal = result_v.x
            sw_v = both_optimal_v * conf_out + visual_optimal * conf_outv
            if both_optimal_v == 0:
                both_optimal_v = 1
            if visual_optimal == 0:
                visual_optimal = 1
            if sw_v < 0:
                sw_v = 128

            print(f"最优分配: a = {both_optimal_v:.4f}, b = {visual_optimal:.4f}")
            print(f"最大社会福利visual: {sw_v:.4f}")

        else:
            both_optimal_v = visual_optimal = 1
            sw_v = both_optimal_v * conf_out + visual_optimal * conf_outv


        # 计算每个类别的概率
        grads_history['both'], grads_history['audio'], grads_history['visual'] = {}, {}, {}

        alpha_both = F.softplus(out) + 1

        alpha_audio = F.softplus(out_a) + 1

        alpha_visual = F.softplus(out_v) + 1

        belief_both = calculate_belief(alpha_both,args.n_classes)
        belief_audio = calculate_belief(alpha_audio,args.n_classes)
        belief_visual = calculate_belief(alpha_visual,args.n_classes)





        for name, param in model.named_parameters():
            if param.grad is not None:
                layer = re.split('[_.]', str(name))
                if ('head' in layer):
                    if 0 <= epoch <= 70 :
                        if epoch <= -1:
                            if ('audio' in layer):
                                if len(param.grad.size()) > 1:
                                    param.grad *= belief_audio.unsqueeze(-1)
                                else:
                                    param.grad *= belief_audio

                            elif ('video' in layer):
                                if len(param.grad.size()) > 1:
                                    param.grad *= belief_visual.unsqueeze(-1)
                                else:
                                    param.grad *= belief_visual

                            else:
                                if len(param.grad.size())>1:
                                    param.grad *= belief_both.unsqueeze(-1)
                                else:
                                    param.grad *= belief_both


                        continue
                if epoch <=100:
                    if ('audio' in layer):
                        if name not in grads_history['both']:
                            grads_history['both'][name] = torch.zeros_like(grads_audio['both'][name],device=param.grad.device)
                            grads_history['audio'][name] = torch.zeros_like(grads_audio['audio'][name],device=param.grad.device)
                        if len(param.grad.size()) == 4:
                            new_grad = 1/both_optimal_a * grads_audio['both'][name] + 1/audio_optimal * grads_audio['audio'][name]
                            param.grad = new_grad * (sw_a / 128)
                        else:
                            param.grad = (grads_audio['both'][name] + grads_audio['audio'][name]) * 1.5




                    if ('visual' in layer):
                        if name not in grads_history['both']:
                            grads_history['both'][name] = torch.zeros_like(grads_visual['both'][name],device=param.grad.device)
                            grads_history['visual'][name] = torch.zeros_like(grads_visual['visual'][name],device=param.grad.device)
                        if len(param.grad.size()) == 4:
                            new_grad = 1/both_optimal_v * grads_visual['both'][name]  + 1/visual_optimal * grads_visual['visual'][
                                name]

                            param.grad = new_grad * (sw_v / 128)

                        else:

                            param.grad = (grads_visual['both'][name] + grads_visual['visual'][name]) * 1.5

        optimizer.step()
        _loss += loss.item()

    return _loss / len(dataloader)


def valid(args, model, device, dataloader):
    n_classes = args.n_classes

    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, images, label) in tqdm(enumerate(dataloader)):

            spec = spec.to(device)
            images = images.to(device)
            label = label.to(device)

            prediction_all = model(spec.float(), images.float())

            prediction = prediction_all[0]
            prediction_audio = prediction_all[1]
            prediction_visual = prediction_all[2]

            for i, item in enumerate(label):

                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)

                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0

                ma_audio = prediction_audio[i].cpu().data.numpy()
                index_ma_audio = np.argmax(ma_audio)
                if index_ma_audio == label[i]:
                    acc_a[label[i]] += 1.0

                ma_visual = prediction_visual[i].cpu().data.numpy()
                index_ma_visual = np.argmax(ma_visual)
                if index_ma_visual == label[i]:
                    acc_v[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')
    model = AVClassifier(args)
    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    if args.dataset == 'CREMAD':
        train_dataset = AVDataset_CD(mode='train')
        test_dataset = AVDataset_CD(mode='test')
        # train_dataset = CramedDataset(args, mode='train')
        # test_dataset = CramedDataset(args, mode='test')
    # elif args.dataset == 'AVE':
    #     train_dataset = AVDataset_AVE(mode='train')
    #     test_dataset = AVDataset_AVE(mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=16, pin_memory=False)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=16)

    if args.optimizer == 'sgd':

        optimizer = optim.SGD(model.parameters(),momentum=0.9, lr=args.learning_rate, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    elif args.optimizer == 'adamgrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    print(len(train_dataloader))

    if args.train:
        best_acc = -1

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            batch_loss = train_epoch(
                args, epoch, model, device, train_dataloader, optimizer, scheduler, None)

            acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'best_model_{}_of_{}_{}_epoch{}_batch{}_lr{}.pth'.format(
                    args.model, args.optimizer, args.dataset, args.epochs, args.batch_size, args.learning_rate)

                saved_dict = {'saved_epoch': epoch,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)

                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.4f}, Acc: {:.4f}, Acc_a: {:.4f}, Acc_v: {:.4f}".format(
                    batch_loss, acc, acc_a, acc_v))
            else:
                print("Loss: {:.4f}, Acc: {:.4f}, Acc_a: {:.4f}, Acc_v: {:.4f},Best Acc: {:.4f}".format(
                    batch_loss, acc, acc_a, acc_v, best_acc))


if __name__ == "__main__":
    main()

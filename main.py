import numpy as np
from sklearn.metrics import auc, roc_curve
import argparse
import load_data
import torch
import torch.nn as nn
import GCN_embedding
from torch.autograd import Variable
from graph_sampler import GraphSampler
from numpy.random import seed
import random
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")


def arg_parse():
    parser = argparse.ArgumentParser(description='GLADD Arguments.')
    parser.add_argument('--datadir', dest='datadir', default='dataset', help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default='DHFR', help='dataset name')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=0,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--clip', dest='clip', default=0.1, type=float, help='Gradient clipping.')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float, help='Learning Rate.')
    parser.add_argument('--num-epochs', dest='num_epochs', default=100, type=int, help='total epoch number')
    parser.add_argument('--batch-size', dest='batch_size', default=800, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=512, type=int, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', default=256, type=int, help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', default=3, type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', default=0.5, type=float, help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    parser.add_argument('--sign', dest='sign', type=int, default=1, help='sign of graph anomaly')
    parser.add_argument('--feature', dest='feature', default='default', help='use what node feature',
                        choices=['default', 'deg-num'])
    parser.add_argument('--train-teacher', dest='train_teacher', default=True, help='Whether trainning the teacher')
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(dataset_p, dataset_n, data_test_loader, model_teacher, model_student1, model_student2, args):
    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model_student1.parameters()), lr=args.lr)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.5)
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model_student2.parameters()), lr=args.lr)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.5)
    max_AUC = 0
    auroc_final = 0
    for epoch in range(args.num_epochs):
        model_student1.train()
        model_student2.train()

        for batch_idx, data in enumerate(dataset_p):
            model_student1.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()

            embed_node, embed = model_student1(h0, adj)
            embed_teacher_node, embed_teacher = model_teacher(h0, adj)
            embed_teacher = embed_teacher.detach()
            embed_teacher_node = embed_teacher_node.detach()
            loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(
                dim=1).mean(dim=0)
            loss = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1).mean(dim=0)
            loss = loss + loss_node
            loss.backward(loss.clone().detach())
            nn.utils.clip_grad_norm_(model_student1.parameters(), args.clip)
            optimizer1.step()
            scheduler1.step()

        for batch_idx, data in enumerate(dataset_n):
            model_student2.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()

            embed_node, embed = model_student2(h0, adj)
            embed_teacher_node, embed_teacher = model_teacher(h0, adj)
            embed_teacher = embed_teacher.detach()
            embed_teacher_node = embed_teacher_node.detach()
            loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(
                dim=1).mean(dim=0)
            loss = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1).mean(dim=0)
            loss = loss + loss_node

            loss.backward(loss.clone().detach())
            nn.utils.clip_grad_norm_(model_student2.parameters(), args.clip)
            optimizer2.step()
            scheduler2.step()

        if (epoch + 1) % 10 == 0 and epoch > 0:
            model_student1.eval()
            model_student2.eval()
            loss = []
            y = []
            mr = []

            for batch_idx, data in enumerate(data_test_loader):
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float(), requires_grad=False).cuda()

                embed_node1, embed1 = model_student1(h0, adj)
                embed_teacher_node, embed_teacher = model_teacher(h0, adj)
                loss_node1 = torch.mean(F.mse_loss(embed_node1, embed_teacher_node, reduction='none'), dim=2).mean(
                    dim=1)
                loss_graph1 = F.mse_loss(embed1, embed_teacher, reduction='none').mean(dim=1)
                loss1 = (loss_node1 + loss_graph1)
                embed_node2, embed2 = model_student2(h0, adj)
                loss_node2 = torch.mean(F.mse_loss(embed_node2, embed_teacher_node, reduction='none'), dim=2).mean(
                    dim=1)
                loss_graph2 = F.mse_loss(embed2, embed_teacher, reduction='none').mean(dim=1)
                loss2 = (loss_node2 + loss_graph2)

                loss.append([loss1.cpu().detach().numpy()[0], loss2.cpu().detach().numpy()[0]])
                if data['label'] == args.sign:
                    y.append(1)
                else:
                    y.append(0)
                mr.append([data['label'].cpu().detach().numpy()[0], loss1.cpu().detach().numpy()[0],
                           loss2.cpu().detach().numpy()[0]])


            label_test = []

            for loss_ in loss:
                label_test.append(loss_[0] - loss_[1])
            label_test = np.array(label_test)

            fpr_ab, tpr_ab, thr_ = roc_curve(y_true=y, y_score=label_test, drop_intermediate=False)
            test_roc_ab = auc(fpr_ab, tpr_ab)
            print('abnormal detection: auroc_ab: {}'.format(test_roc_ab))
            if test_roc_ab > max_AUC:
                max_AUC = test_roc_ab
        if epoch == (args.num_epochs - 1):
            auroc_final = max_AUC

    return auroc_final


def train_teacher(dataset, model_teacher, args):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_teacher.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    for epoch in range(args.num_epochs):
        model_teacher.train()

        for batch_idx, data in enumerate(dataset):
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            embed_teacher_node, embed_teacher = model_teacher(h0, adj)
            loss = embed_teacher.std(dim=0).mean(dim=0)
            loss_node = embed_teacher_node.std(dim=1).mean(dim=1).mean(dim=0)
            loss = 1 / (loss + loss_node)
            loss.backward(loss.clone().detach())
            nn.utils.clip_grad_norm_(model_teacher.parameters(), args.clip)
            optimizer.step()
            scheduler.step()


if __name__ == '__main__':
    args = arg_parse()
    setup_seed(args.seed)
    large = not args.DS.find("Tox21_") == -1

    graphs = load_data.read_graphfile(args.datadir, args.DS, max_nodes=args.max_nodes)
    datanum = len(graphs)
    if args.max_nodes == 0:
        max_nodes_num = max([G.number_of_nodes() for G in graphs])
    else:
        max_nodes_num = args.max_nodes
    print('GraphNumber: {}'.format(datanum))
    graphs_label = [graph.graph['label'] for graph in graphs]

    if large:
        DST = args.DS[:args.DS.rfind('_')] + "_testing"
        graphs_testgroup = load_data.read_graphfile(args.datadir, DST, max_nodes=args.max_nodes)
        datanum_test = len(graphs_testgroup)
        if args.max_nodes == 0:
            max_nodes_num = max([max([G.number_of_nodes() for G in graphs_testgroup]), max_nodes_num])
        else:
            max_nodes_num = args.max_nodes
        graphs_label_test = [graph.graph['label'] for graph in graphs_testgroup]

        graphs_all = graphs + graphs_testgroup
        graphs_label_all = graphs_label + graphs_label_test
    else:
        graphs_all = graphs
        graphs_label_all = graphs_label

    kfd = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
    result_auc = []
    for k, (train_index, test_index) in enumerate(kfd.split(graphs_all, graphs_label_all)):
        graphs_train_ = [graphs_all[i] for i in train_index]
        graphs_test = [graphs_all[i] for i in test_index]

        graphs_train_n = []
        graphs_train_p = []
        for graph in graphs_train_:
            if graph.graph['label'] != args.sign:
                graphs_train_p.append(graph)
            else:
                graphs_train_n.append(graph)

        num_train_p = len(graphs_train_p)
        num_train_n = len(graphs_train_n)
        num_test = len(graphs_test)
        print('TrainSize_p: {}, TrainSize_n: {}, TestSize: {}'.format(num_train_p, num_train_n, num_test))

        dataset_sampler_train = GraphSampler(graphs_train_, features=args.feature, normalize=False,
                                             max_num_nodes=max_nodes_num)
        dataset_sampler_train_p = GraphSampler(graphs_train_p, features=args.feature, normalize=False,
                                               max_num_nodes=max_nodes_num)
        dataset_sampler_train_n = GraphSampler(graphs_train_n, features=args.feature, normalize=False,
                                               max_num_nodes=max_nodes_num)

        model_teacher = GCN_embedding.GcnEncoderGraph_teacher(dataset_sampler_train_p.feat_dim, args.hidden_dim,
                                                              args.output_dim, 2,
                                                              args.num_gc_layers, bn=args.bn, args=args).cuda()
        model_student1 = GCN_embedding.GcnEncoderGraph_teacher(dataset_sampler_train_p.feat_dim,
                                                               args.hidden_dim,
                                                               args.output_dim, 2,
                                                               args.num_gc_layers, bn=args.bn,
                                                               args=args).cuda()
        model_student2 = GCN_embedding.GcnEncoderGraph_teacher(dataset_sampler_train_p.feat_dim,
                                                               args.hidden_dim,
                                                               args.output_dim, 2,
                                                               args.num_gc_layers, bn=args.bn,
                                                               args=args).cuda()

        data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train,
                                                        shuffle=True,
                                                        batch_size=args.batch_size)
        data_train_loader_p = torch.utils.data.DataLoader(dataset_sampler_train_p,
                                                          shuffle=True,
                                                          batch_size=args.batch_size)
        data_train_loader_n = torch.utils.data.DataLoader(dataset_sampler_train_n,
                                                          shuffle=True,
                                                          batch_size=args.batch_size)

        dataset_sampler_test = GraphSampler(graphs_test, features=args.feature, normalize=False,
                                            max_num_nodes=max_nodes_num)
        data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test,
                                                       shuffle=False,
                                                       batch_size=1)
        if args.train_teacher:
            train_teacher(data_train_loader, model_teacher, args)

        result = train(data_train_loader_p, data_train_loader_n, data_test_loader, model_teacher, model_student1,
                       model_student2, args)

        result_auc.append(result)

    result_auc = np.array(result_auc)
    auc_avg = np.mean(result_auc)
    auc_std = np.std(result_auc)
    print('auroc{}, average: {}, std: {}'.format(result_auc, auc_avg, auc_std))

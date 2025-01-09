import torch
import argparse
import time

from data import get_data
from models import *
from baselines import *


def build_model(args, edge_index, num_nodes, num_features, num_classes, device):
    if args.model == 'Neural_Simplified':
        model = DIGNN_Neural_Simplified(in_channels=num_features,
                                    out_channels=num_classes,
                                    hidden_channels=args.num_hid,
                                    edge_index=edge_index,
                                    num_nodes=num_nodes,
                                    device=device,
                                    mu=args.mu,
                                    max_iter=args.max_iter,
                                    threshold=args.threshold,
                                    dropout=args.dropout,
                                    preprocess=args.preprocess)
    elif args.model == 'Neural_Simplified_L1':
        model = DIGNN_Neural_Simplified_L1(in_channels=num_features,
                                    out_channels=num_classes,
                                    hidden_channels=args.num_hid,
                                    edge_index=edge_index,
                                    num_nodes=num_nodes,
                                    device=device,
                                    dropout=args.dropout,
                                    rho=args.rho,
                                    lbd = args.lbd,
                                    mu_k_max=args.mu_k_max,
                                    mu_k=args.mu_k,
                                    K_iter=args.K_iter,
                                    preprocess=args.preprocess)

    elif args.model == 'Neural':
        model = DIGNN_Neural(in_channels=num_features,
                             out_channels=num_classes,
                             hidden_channels=args.num_hid,
                             edge_index=edge_index,
                             num_nodes=num_nodes,
                             device=device,
                             mu=args.mu,
                             max_iter=args.max_iter,
                             threshold=args.threshold,
                             dropout=args.dropout,
                             preprocess=args.preprocess)
    elif args.model == 'mlp':
        model = MLPNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'gcn':
        model = GCNNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'sgc':
        model = SGCNet(in_channels=num_features,
                        out_channels=num_classes,
                        K=args.K)
    elif args.model == 'gat':
        model = GATNet(in_channels=num_features,
                       out_channels=num_classes,
                       num_hid=args.num_hid,
                       num_heads=args.num_heads,
                       dropout=args.dropout)
    elif args.model == 'jk':
        model = JKNet(in_channels=num_features,
                      out_channels=num_classes,
                      num_hid=args.num_hid,
                      K=args.K,
                      alpha=args.alpha,
                      dropout=args.dropout)
    elif args.model == 'appnp':
        model = APPNPNet(in_channels=num_features,
                         out_channels=num_classes,
                         num_hid=args.num_hid,
                         K=args.K,
                         alpha=args.alpha,
                         dropout=args.dropout)
    elif args.model == 'gcnii':
        model = GCNIINet(in_channels=num_features,
                         out_channels=num_classes,
                         hidden_channels=args.num_hid,
                         num_layers=args.num_layers,
                         dropout=args.dropout,
                         alpha=args.alpha,
                         theta=args.theta)
    elif args.model == 'h2gcn':
        model = H2GCNNet(in_channels=num_features,
                         out_channels=num_classes,
                         hidden_channels=args.num_hid,
                         edge_index=edge_index,
                         num_layers=args.num_layers,
                         dropout=args.dropout)

    elif args.model =='ignn':
        model = IGNNNet(nfeat=num_features, 
                        nhid=args.num_hid, 
                        nclass=num_classes, 
                        num_node=num_nodes, 
                        dropout=args.dropout, 
                        kappa=0.9, 
                        adj_orig=None)
        
    elif args.model == 'RD_GCN':
        model = RD_GCN(in_channels=num_features, 
                       out_channels=num_classes, 
                       num_hid=args.num_hid,
                       dropout=args.dropout,
                       rho=args.rho,
                       mu_k_max=args.mu_k_max,
                       mu_k = args.mu_k,
                       K_iter = args.K_iter)
    
    elif args.model == 'RD_GAT':
        model = RD_GAT(in_channels=num_features, 
                       out_channels=num_classes, 
                       num_hid=args.num_hid,
                       num_heads=args.num_heads,
                       dropout=args.dropout,
                       rho=args.rho,
                       mu_k_max=args.mu_k_max,
                       mu_k = args.mu_k,
                       K_iter = args.K_iter)
    
    elif args.model == 'RD_APPNP':
        model = RD_APPNP(in_channels=num_features, 
                         out_channels=num_classes,
                         num_hid=args.num_hid,
                         K=args.K,
                         alpha=args.alpha,
                         dropout=args.dropout,
                         rho=args.rho,
                         mu_k_max=args.mu_k_max,
                         mu_k = args.mu_k,
                         K_iter = args.K_iter)

    return model

def train(model, optimizer, x, y, edge_index, train_mask):
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index)
    loss = F.nll_loss(output[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(model, x, y, edge_index, train_mask, val_mask, test_mask):
    model.eval()
    logits, accs = model(x, edge_index), []
    train_pred = logits[train_mask].max(1)[1]
    train_acc = train_pred.eq(y[train_mask]).sum().item() / train_mask.sum().item()
    accs.append(train_acc)
    val_pred = logits[val_mask].max(1)[1]
    val_acc = val_pred.eq(y[val_mask]).sum().item() / val_mask.sum().item()
    accs.append(val_acc)
    test_pred = logits[test_mask].max(1)[1]
    test_acc = test_pred.eq(y[test_mask]).sum().item() / test_mask.sum().item()
    accs.append(test_acc)
    return accs

def add_noise_features(features_perturb, base_ratio, extreme_ratio, dataset_name):

    num_nodes, num_feats = features_perturb.shape
    dataset_name = dataset_name.lower()
    total_elements = num_nodes * num_feats
    num_perturb = int(total_elements * base_ratio)

    if dataset_name in ['cora', 'citeseer', 'chameleon', 'squirrel']:
        rand_indices_base = torch.randperm(total_elements)[:num_perturb].to(features_perturb.device)
        features_perturb.view(-1)[rand_indices_base] = 1 - features_perturb.view(-1)[rand_indices_base]

        if extreme_ratio > 0.0:
            num_extreme_noise = int(total_elements * extreme_ratio)
            rand_indices_extreme = torch.randperm(total_elements)[:num_extreme_noise].to(features_perturb.device)
            extreme_noise_to_add = torch.randint(low=0, high=15, size=(num_extreme_noise,)).to(features_perturb.device)
            features_perturb.view(-1)[rand_indices_extreme] = extreme_noise_to_add.float()

    elif dataset_name in ['pubmed', 'penn94', 'cornell5', 'amherst41']:
        rand_index_base = torch.normal(mean=0, std=0.1, size=features_perturb.shape).to(features_perturb.device)
        features_perturb += rand_index_base

        if extreme_ratio > 0.0:
            num_extreme_noise = int(total_elements * extreme_ratio)
            rand_indices_extreme = torch.randperm(total_elements)[:num_extreme_noise].to(features_perturb.device)
            extreme_noise_to_add = (torch.randint(low=0, high=1, size=(num_extreme_noise,)).to(features_perturb.device) * 2 - 1) * 20.
            features_perturb.view(-1)[rand_indices_extreme] += extreme_noise_to_add
    return features_perturb



def main(args):
    #print(args)
    print(args.input, args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    edge_index, features, labels, in_channels, out_channels, \
    train_mask, val_mask, test_mask = get_data('data', args.input, device)

    if args.add_noise:
        features = add_noise_features(features, args.base_ratio, args.extreme_ratio, args.input)

    num_nodes = features.size(0)
    results = []
    times_used = []

    for run in range(args.runs):
        idx_train, idx_val, idx_test = train_mask[:, run], val_mask[:, run], test_mask[:, run]
        
        model = build_model(args, edge_index, num_nodes, in_channels, out_channels, device)
        model = model.to(device)
        # data = data.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        t1 = time.time()
        best_val_acc = test_acc = 0
        for epoch in range(1, args.epochs+1):
            # t01 = time.time()
            train(model, optimizer, features, labels, edge_index, idx_train)
            # t02 = time.time()
            train_acc, val_acc, tmp_test_acc = test(model, features, labels, edge_index, idx_train, idx_val, idx_test)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
            # print(t02 - t01)
        t2 = time.time()
        print('{}, {}, Run: {:02d}, Accuracy: {:.4f}, Time: {:.4f}'.format(args.model, args.input, run+1, test_acc, t2-t1))
        results.append(test_acc)
        times_used.append(t2-t1)
    results = 100 * torch.Tensor(results)
    times_used = np.array(times_used)
    print(results)
    print(times_used)
    print(f'Averaged test accuracy for {args.runs} runs: {results.mean():.2f} \pm {results.std():.2f}')
    print(f'Averaged time used for {args.runs} runs: {times_used.mean():.2f} \pm {times_used.std():.2f}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        type=str, 
                        default='chameleon',   
                        choices=['chameleon', 'squirrel', 'penn94', 'cornell5', 'amherst41', 'cora', 'citeseer', 'pubmed'],
                        help='Input graph.')
    parser.add_argument('--train_rate', 
                        type=float, 
                        default=0.05,
                        help='Training rate.')
    parser.add_argument('--val_rate', 
                        type=float, 
                        default=0.05,
                        help='Validation rate.')
    parser.add_argument('--model',
                        type=str,
                        default='RD_GCN',
                        choices=['mlp', 'gcn', 'gat', 'jk', 'appnp', 'gcnii', 'h2gcn', 'gind', 'ignn', 'Neural',
                                 'Neural_Simplified', 'Neural_Simplified_L1', 'RD_GCN', 'RD_GAT', 'RD_APPNP'],
                        help='GNN model')
    parser.add_argument('--runs',
                        type=int,
                        default=5,
                        help='Number of repeating experiments.')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', 
                        type=float, 
                        default=0.001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--num_hid', 
                        type=int, 
                        default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.5,
                        help='Dropout rate (1 - keep probability).')
    
    # GAT
    parser.add_argument('--num_heads', 
                        type=int, 
                        default=8,
                        help='Number of heads.')
    
    # SGC & APPNP
    parser.add_argument('--K', 
                        type=int, 
                        default=4,
                        help='K.')
    
    # APPNP & GCNII
    parser.add_argument('--alpha', 
                        type=float, 
                        default=0.5,
                        help='alpha.')
    
    # GCNII
    parser.add_argument('--theta',
                        type=float,
                        default=1.,
                        help='theta.')
    
    # GCNII & H2GCN
    parser.add_argument('--num_layers', 
                        type=int, 
                        default=1,
                        help='Number of layers.')
    
    # Implicit setting
    parser.add_argument('--max_iter',
                        type=int,
                        default=10)
    parser.add_argument('--threshold',
                        type=float,
                        default=1e-6)

    parser.add_argument('--preprocess',
                        type=str,
                        default='adj')

    # DIGNN and F-DIGNN
    parser.add_argument('--mu',
                        type=float,
                        default=2.2,
                        help='mu.')
    
    # ADMM and L1-DIGNN
    parser.add_argument('--lbd',
                        type=float,
                        default=1.0,
                        help='lambda value.')
    parser.add_argument('--rho',
                        type=float,
                        default=1.1,
                        help='rho.')
    parser.add_argument('--mu_k_max',
                        type=float,
                        default=1000000,
                        help='mu_k_max.')
    parser.add_argument('--mu_k',
                        type=float,
                        default=1.,
                        help='mu_k.')
    parser.add_argument('--K_iter',
                        type=int,
                        default=6,
                        help='num iter for ADMM.')
    
    # Noise setting
    parser.add_argument('--add_noise',
                        type=bool,
                        default=False,
                        help='add_noise')
    parser.add_argument('--base_ratio',
                        type=float,
                        default=0.0,
                        help='base_ratio')
    parser.add_argument('--extreme_ratio',
                        type=float,
                        default=0.0,
                        help='extreme_ratio')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main(get_args())



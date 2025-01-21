# -*- coding = utf-8 -*-
from ast import mod
import numpy as np
import torch
import copy
import time
import hdbscan
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from models.Update import LocalUpdate
import heapq
import torch.nn.functional as F
import kmeans1d
from sklearn.cluster import AgglomerativeClustering,KMeans,DBSCAN
from torch.utils.data import DataLoader, Dataset,Subset
def cos(a, b):
    res = (np.dot(a, b) + 1e-9) / (np.linalg.norm(a) + 1e-9) / \
        (np.linalg.norm(b) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res


def fltrust(params, central_param, global_parameters, args):
    FLTrustTotalScore = 0
    score_list = []
    central_param_v = parameters_dict_to_vector_flt(central_param)
    central_norm = torch.norm(central_param_v)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    sum_parameters = None
    for local_parameters in params:
        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)
        client_cos = cos(central_param_v, local_parameters_v)
        client_cos = max(client_cos.item(), 0)
        client_clipped_value = central_norm/torch.norm(local_parameters_v)
        score_list.append(client_cos)
        FLTrustTotalScore += client_cos
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in local_parameters.items():
                sum_parameters[key] = client_cos * \
                    client_clipped_value * var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * local_parameters[
                    var]
    if FLTrustTotalScore == 0:
        print(score_list)
        return global_parameters
    for var in global_parameters:
        temp = (sum_parameters[var] / FLTrustTotalScore)
        if global_parameters[var].type() != temp.type():
            temp = temp.type(global_parameters[var].type())
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
        else:
            global_parameters[var] += temp * args.server_lr
    print(score_list)
    return global_parameters


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector_flt_cpu(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        vec.append(param.cpu().view(-1))
    return torch.cat(vec)


def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters


import torch

def kernel_function(x, y):
    sigma = 1.0
    return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))

def compute_mmd(x, y):
    # Compute the MMD between two tensors x and y
    # x and y should have the same number of samples
    m = x.size(0)
    n = y.size(0)
    # Compute the kernel matrices for x and y
    xx_kernel = torch.zeros((m, m))
    yy_kernel = torch.zeros((n, n))
    xy_kernel = torch.zeros((m, n))
    for i in range(m):
        for j in range(i, m):
            xx_kernel[i, j] = xx_kernel[j, i] = kernel_function(x[i], x[j])

    for i in range(n):
        for j in range(i, n):
            yy_kernel[i, j] = yy_kernel[j, i] = kernel_function(y[i], y[j])

    for i in range(m):
        for j in range(n):
            xy_kernel[i, j] = kernel_function(x[i], y[j])
    # Compute the MMD statistic
    mmd = (torch.sum(xx_kernel) / (m * (m - 1))) + (torch.sum(yy_kernel) / (n * (n - 1))) - (2 * torch.sum(xy_kernel) / (m * n))
    return mmd


def flare(w_updates, w_locals, net, central_dataset, dataset_test, global_parameters, args):
    w_feature=[]
    temp_model = copy.deepcopy(net)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    
    for client in w_locals:
        net.load_state_dict(client)
        local = LocalUpdate(
                args=args, dataset=dataset_test, idxs=central_dataset)
        feature = local.get_PLR(
            net=copy.deepcopy(net).to(args.device))
        w_feature.append(feature)
    distance_list=[[] for i in range(len(w_updates))]
    # distance_list=[list(len(w_updates)) for i in range(len(w_updates))]
    for i in range(len(w_updates)):
        for j in range(i+1, len(w_updates)):
            score = compute_mmd(w_feature[i], w_feature[j])
            distance_list[i].append(score.item())
            distance_list[j].append(score.item())
    print('defense line121 distance_list', distance_list)
    vote_counter=[0 for i in range(len(w_updates))]
    k = round(len(w_updates)*0.5)
    for i in range(len(w_updates)):
        IDs = np.argsort(distance_list[i])
        for j in range(len(IDs)):
            # client_id is the index of client i-th client voting for
            # distance_list[] only records score with other clients without itself
            # so distance_list[i][i] should be itself
            # client_id = j + 1 after j >= i
            if IDs[j] >= i:
                client_id = IDs[j] + 1 
            else:
                client_id = IDs[j]
            vote_counter[client_id] += 1
            if j + 1 >= k:  # first 𝑘 elements in 𝐼 𝐷𝑠 and vote for it
                break

    trust_score = [x/sum(vote_counter) for x in vote_counter]
    # print('defense line188 len trust_score', trust_score)
    
    w_avg = copy.deepcopy(global_parameters)
    for k in w_avg.keys():
        for i in range(0, len(w_updates)):
            try:
                w_avg[k] += w_updates[i][k] * trust_score[i]
            except:
                print("Fed.py line17 type_as", 'w_updates[i][k].type():', w_updates[i][k].type(), k)
                w_updates[i][k] = w_updates[i][k].type_as(w_avg[k]).long()
                w_avg[k] = w_avg[k].long() + w_updates[i][k] * trust_score[i]
    return w_avg


def log_layer_wise_distance(updates):
    # {layer_name, [layer_distance1, layer_distance12...]}
    layer_distance = {}
    for layer, val in updates[0].items():
        if 'num_batches_tracked' in layer:
            continue
        # for each layer calculate distance among models
        for model in updates:
            temp_layer_dis = 0
            for model2 in updates:
                temp_norm = torch.norm((model[layer] - model2[layer]))
                temp_layer_dis += temp_norm
            if layer not in layer_distance.keys():
                layer_distance[layer] = []
            layer_distance[layer].append(temp_layer_dis.item())
    return layer_distance
    
        
def layer_krum(gradients, n_attackers, args, multi_k=False):
    new_global = {}
    for layer in gradients[0].keys():
        if layer.split('.')[-1] == 'num_batches_tracked' or layer.split('.')[-1] == 'running_mean' or layer.split('.')[-1] == 'running_var':
            new_global[layer] = gradients[-1][layer]
        else:
            layer_gradients = [x[layer] for x in gradients]
            new_global[layer] = layer_multi_krum(layer_gradients, n_attackers, args, multi_k)
    return new_global

def layer_flatten_grads(gradients):
    flat_epochs = []
    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        flat_epochs.append(gradients[n_user].cpu().numpy().flatten().tolist())
    flat_epochs = np.array(flat_epochs)
    return flat_epochs

def layer_multi_krum(layer_gradients, n_attackers, args, multi_k=False):
    grads = layer_flatten_grads(layer_gradients)
    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))
    score_record = None
    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        scores = None
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)
        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        if args.log_distance == True and score_record == None:
            print('defense.py line149 (krum distance scores):', scores)
            score_record = scores
            args.krum_distance.append(scores)
            layer_distance_dict = log_layer_wise_distance(gradients)
            args.krum_layer_distance.append(layer_distance_dict)
            # print('defense.py line149 (layer_distance_dict):', layer_distance_dict)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    agg_layer = 0
    for selected_layer in candidate_indices:
        agg_layer += layer_gradients[selected_layer]
    agg_layer /= len(candidate_indices)
    return agg_layer

def multi_krum(gradients, n_attackers, args, multi_k=False):
    grads = flatten_grads(gradients)
    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))
    
    score_record = None

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        
        if args.log_distance == True and score_record == None:
            print('defense.py line149 (krum distance scores):', scores)
            score_record = scores
            args.krum_distance.append(scores)
            layer_distance_dict = log_layer_wise_distance(gradients)
            args.krum_layer_distance.append(layer_distance_dict)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    args.turn+=1
    for selected_client in candidate_indices:
        if selected_client < num_malicious_clients:
            args.wrong_mal += 1

    for i in range(len(scores)):
        if i < num_malicious_clients:
            args.mal_score += scores[i]
        else:
            args.ben_score += scores[i]
    
    return np.array(candidate_indices)



def flatten_grads(gradients):

    param_order = gradients[0].keys()

    flat_epochs = []

    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in param_order:
            try:
                user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
            except:
                user_arr.extend(
                    [grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs




def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        update2[key] = update[key] - model[key]
    return update2

def get_update2(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        update2[key] = update[key] - model[key]
    return update2


def fld_distance(old_update_list, local_update_list, net_glob, attack_number, hvp):
    pred_update = []
    distance = []
    for i in range(len(old_update_list)):
        pred_update.append((old_update_list[i] + hvp).view(-1))
        
    
    pred_update = torch.stack(pred_update)
    local_update_list = torch.stack(local_update_list)
    old_update_list = torch.stack(old_update_list)
    
    distance = torch.norm((old_update_list - local_update_list), dim=1)
    print('defense line219 distance(old_update_list - local_update_list):',distance)

    distance = torch.norm((pred_update - local_update_list), dim=1)
    distance = distance / torch.sum(distance)
    return distance

def detection(score, nobyz):
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    
    if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
        #0 is the label of malicious clients
        label_pred = 1 - label_pred
    real_label=np.ones(100)
    real_label[:nobyz]=0
    acc=len(label_pred[label_pred==real_label])/100
    recall=1-np.sum(label_pred[:nobyz])/nobyz
    fpr=1-np.sum(label_pred[nobyz:])/(100-nobyz)
    fnr=np.sum(label_pred[:nobyz])/nobyz
    # print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
    # print(silhouette_score(score.reshape(-1, 1), label_pred))
    print('defence.py line233 label_pred (0 = malicious pred)', label_pred)
    return label_pred

def detection1(score):
    nrefs = 10
    ks = range(1, 8)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
    score = (score - min)/(max-min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m]-center[label_pred[m]]) for m in range(len(score))])
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m]-center[label_pred[m]]) for m in range(len(rand))])
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    select_k = 2  # default detect attacks
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i+1
            break
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        print('Attack Detected!')
        return 1

def RLR(global_model, agent_updates_list, args):
    """
    agent_updates_dict: dict['key']=one_dimension_update
    agent_updates_list: list[0] = model.dict
    global_model: net
    """
    args.server_lr = 1

    grad_list = []
    for i in agent_updates_list:
        grad_list.append(parameters_dict_to_vector_rlr(i))
    agent_updates_list = grad_list
    

    aggregated_updates = 0
    for update in agent_updates_list:
        # print(update.shape)  # torch.Size([1199882])
        aggregated_updates += update
    aggregated_updates /= len(agent_updates_list)
    lr_vector = compute_robustLR(agent_updates_list, args)
    cur_global_params = parameters_dict_to_vector_rlr(global_model.state_dict())
    print('defense.py line 430 lr_vector == -1', lr_vector[lr_vector==-1].shape[0]/lr_vector.view(-1).shape[0])
    new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
    global_w = vector_to_parameters_dict(new_global_params, global_model.state_dict())
    # print(cur_global_params == vector_to_parameters_dict(new_global_params, global_model.state_dict()))
    return global_w

def parameters_dict_to_vector_rlr(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)



def vector_to_parameters_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict

def compute_robustLR(params, args):
    agent_updates_sign = [torch.sign(update) for update in params]  
    sm_of_signs = torch.abs(sum(agent_updates_sign))
    # print(len(agent_updates_sign)) #10
    # print(agent_updates_sign[0].shape) #torch.Size([1199882])
    sm_of_signs[sm_of_signs < args.robustLR_threshold] = -args.server_lr
    sm_of_signs[sm_of_signs >= args.robustLR_threshold] = args.server_lr 
    return sm_of_signs.to(args.gpu)
   
    


def flame(local_model, update_params, global_model, args, debug=False):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    if debug==True:
        filename = './' + args.save + '/flame_analysis.txt'
        f = open(filename, "a")
        for i in cos_list:
            f.write(str(i))
            print(i)
            f.write('\n')
        f.write('\n')
        f.write("--------Round--------")
        f.write('\n')
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
                norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    print(benign_client)
   
    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal+=1
        else:
            #  minus per benign in cluster
            args.right_ben += 1
    args.turn+=1

    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value/norm_list[i]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
    #add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
                    continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0,std=args.noise*clip_value)
        var += temp
    return global_model


def flame_analysis(local_model, args, debug=False):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    if debug==True:
        filename = './' + args.save + '/flame_analysis.txt'
        f = open(filename, "a")
        for i in cos_list:
            f.write(str(i))
            f.write('/n')
        f.write('/n')
        f.write("--------Round--------")
        f.write('/n')
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    return benign_client

def lbfgs(args, S_k_list, Y_k_list, v):
    curr_S_k = nd.concat(*S_k_list, dim=1)
    curr_Y_k = nd.concat(*Y_k_list, dim=1)
    S_k_time_Y_k = nd.dot(curr_S_k.T, curr_Y_k)
    S_k_time_S_k = nd.dot(curr_S_k.T, curr_S_k)
    R_k = np.triu(S_k_time_Y_k.asnumpy())
    L_k = S_k_time_Y_k - nd.array(R_k, ctx=mx.gpu(args.gpu))
    sigma_k = nd.dot(Y_k_list[-1].T, S_k_list[-1]) / (nd.dot(S_k_list[-1].T, S_k_list[-1]))
    D_k_diag = nd.diag(S_k_time_Y_k)
    upper_mat = nd.concat(*[sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = nd.concat(*[L_k.T, -nd.diag(D_k_diag)], dim=1)
    mat = nd.concat(*[upper_mat, lower_mat], dim=0)
    mat_inv = nd.linalg.inverse(mat)

    approx_prod = sigma_k * v
    p_mat = nd.concat(*[nd.dot(curr_S_k.T, sigma_k * v), nd.dot(curr_Y_k.T, v)], dim=0)
    approx_prod -= nd.dot(nd.dot(nd.concat(*[sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)

    return approx_prod


def lbfgs_torch(args, S_k_list, Y_k_list, v):
    curr_S_k = torch.stack(S_k_list)
    curr_S_k = curr_S_k.transpose(0, 1).cpu() #(10,xxxxxx)
    print('------------------------')
    print('curr_S_k.shape', curr_S_k.shape)
    curr_Y_k = torch.stack(Y_k_list)
    curr_Y_k = curr_Y_k.transpose(0, 1).cpu() #(10,xxxxxx)
    S_k_time_Y_k = curr_S_k.transpose(0, 1) @ curr_Y_k
    S_k_time_Y_k = S_k_time_Y_k.cpu()


    S_k_time_S_k = curr_S_k.transpose(0, 1) @ curr_S_k
    S_k_time_S_k = S_k_time_S_k.cpu()
    print('S_k_time_S_k.shape', S_k_time_S_k.shape)
    R_k = np.triu(S_k_time_Y_k.numpy())
    L_k = S_k_time_Y_k - torch.from_numpy(R_k).cpu()
    sigma_k = Y_k_list[-1].view(-1,1).transpose(0, 1) @ S_k_list[-1].view(-1,1) / (S_k_list[-1].view(-1,1).transpose(0, 1) @ S_k_list[-1].view(-1,1))
    sigma_k=sigma_k.cpu()
    
    D_k_diag = S_k_time_Y_k.diagonal()
    upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = torch.cat([L_k.transpose(0, 1), -D_k_diag.diag()], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat_inv = mat.inverse()
    print('mat_inv.shape',mat_inv.shape)
    v = v.view(-1,1).cpu()

    approx_prod = sigma_k * v
    print('approx_prod.shape',approx_prod.shape)
    print('v.shape',v.shape)
    print('sigma_k.shape',sigma_k.shape)
    print('sigma_k',sigma_k)
    p_mat = torch.cat([curr_S_k.transpose(0, 1) @ (sigma_k * v), curr_Y_k.transpose(0, 1) @ v], dim=0)
    
    approx_prod -= torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1) @ mat_inv @ p_mat
    print('approx_prod.shape',approx_prod.shape)

    return approx_prod.T


class DefenseMethods:  
    def __init__(self, args, local_data, iters,temp=1):  
        self.args = args  
        self.temp = temp  
        self.local_data = local_data  
        self.iters = iters  
        #self.attackers = attacker_users  
        self.reset_state()  
        #self.class_weights = self.calculate_class_weights()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def reset_state(self):  
        self.global_score = []  
        self.clusters = []  
        self.centroids = []  
        self.metric_honests = []  
        self.metric_maliciouses = []  

    def Clustering(self, metrics):  
        self.reset_state()  
        self.global_score = [0] * len(metrics)  
        self.clusters, self.centroids = kmeans1d.cluster(metrics, 2)  
        
        # if not self.clusters:  
        #     return [], 0  

        honest = max(set(self.clusters), key=self.clusters.count)  
        malicious = min(set(self.clusters), key=self.clusters.count)
        # self.reset_state()  
        # self.global_score = [0] * len(metrics)  
        
        # Use DBSCAN for more robust clustering  
        # clustering = DBSCAN(eps=0.3, min_samples=3).fit(np.array(metrics).reshape(-1, 1))  
        # labels = clustering.labels_  
        
        #if len(set(labels)) < 2:  
            # Fallback to kmeans1d if DBSCAN fails  
        #self.clusters, self.centroids = kmeans1d.cluster(metrics, 2)  
        # else:  
        #     self.clusters = labels  
        #     self.centroids = [np.mean(np.array(metrics)[labels == i]) for i in set(labels) if i != -1]  
        
        # if not self.clusters:  
        #     return [], 0  

        #honest = max(set(self.clusters), key=self.clusters.count)   

        for i, cluster in enumerate(self.clusters):  
            if cluster == honest:  
                self.metric_honests.append(metrics[i])  
                self.global_score[i] = 1  
            else:  
                self.metric_maliciouses.append(metrics[i])  

        honest_cluster_radius = sum(abs(m - self.centroids[honest]) for m in self.metric_honests) / len(self.metric_honests)  
        malicious_cluster_radius = sum(abs(m - self.centroids[malicious]) for m in self.metric_maliciouses) / len(self.metric_maliciouses)  
        cluster_distance = abs(self.centroids[honest] - self.centroids[malicious])
        # honest_cluster_radius = np.std(self.metric_honests) if self.metric_honests else 0  
        # malicious_cluster_radius = np.std(self.metric_maliciouses) if self.metric_maliciouses else 0  
        # cluster_distance = np.abs(np.mean(self.metric_honests) - np.mean(self.metric_maliciouses))    

        return self.global_score, (honest_cluster_radius + malicious_cluster_radius) / (cluster_distance + 1e-8)  

    @staticmethod  
    def Calculate(*lists):  
        filtered_lists = [lst for lst in lists if any(lst)]  
        return [int(all(bits)) for bits in zip(*filtered_lists)] if filtered_lists else []
    
    
    def confidence_weight(self, probs):  
        # 基于预测的置信度计算权重  
        max_probs, _ = probs.max(dim=1)  
        return torch.exp(max_probs - 1)  # 将范围映射到 (0, 1]  
    
    def test(self, G_net, net, ldr_test):  
        # loss_func = nn.KLDivLoss(reduction='batchmean')  
        # device = next(G_net.parameters()).device  

        # loss_sum = cos_dis_sum = total_samples = 0.0  

        # with torch.no_grad():  
        #     for images, labels in ldr_test:  
        #         images, labels = images.to(device), labels.to(device)  
                
        #         G_probs = G_net(images)  
        #         probs = net(images)  
                
        #         cos_dis = F.cosine_similarity(G_probs, probs, dim=1).mean()  
        #         cos_dis_sum += cos_dis.item() * images.size(0)  
                
        #         G_probs = F.softmax(G_probs / self.temp, dim=1)  
        #         probs = F.log_softmax(probs / self.temp, dim=1)  
                
        #         mask = (G_probs.argmax(dim=1) == labels)  
                
        #         if mask.any():  
        #             G_probs_filtered = G_probs[mask]  
        #             probs_filtered = probs[mask]  
        #             loss = loss_func(probs_filtered, G_probs_filtered)  
        #             loss_sum += loss.item() * mask.sum().item()  
                
        #         total_samples += mask.sum().item()
        #device = next(G_net.parameters()).device  
        loss_sum = 0.0  
        total_samples = len(ldr_test.dataset)  
        filtered_samples = 0  

        with torch.no_grad():  
            for images, labels in ldr_test:  
                images, labels = images.to(self.device), labels.to(self.device)  
                
                G_probs = G_net(images)  
                probs = net(images)  
                
                G_probs = F.softmax(G_probs / self.temp, dim=1)  
                probs = F.log_softmax(probs / self.temp, dim=1)  
                
                mask = (G_probs.argmax(dim=1) == labels)  
                
                if mask.any():  
                    G_probs_filtered = G_probs[mask]  
                    probs_filtered = probs[mask]  
                    conf_weight = self.confidence_weight(G_probs_filtered)
                    kl_div = F.kl_div(probs_filtered, G_probs_filtered, reduction='none').sum(dim=1)
                    weighted_kl = (kl_div * conf_weight).mean()  
                    #loss = kl_div.mean()  
                    loss_sum += weighted_kl.item() * mask.sum().item()  
                    filtered_samples += mask.sum().item()  

        if filtered_samples == 0:  
            return 0  # 避免除以零  

        # 计算加权因子  
        weight = (total_samples / filtered_samples)**0.5 
        
        return loss_sum / filtered_samples * weight 

        # return (loss_sum / total_samples if total_samples > 0 else 0,  
        #         cos_dis_sum / total_samples if total_samples > 0 else 0) 
 

    def Adaptive_Agg(self, updates, baseline): 
        cluters_time_start = time.perf_counter() 
        model = copy.deepcopy(baseline).to(self.device)
        global_vector = torch.nn.utils.parameters_to_vector(baseline.parameters())
        model_dicts = []

        for update in updates:
            # 创建模型副本并加载状态字典
            model_copy = copy.deepcopy(baseline).to(self.device)
            model_copy.load_state_dict(update)
            model_dicts.append(model_copy)

        # 将模型参数向量化
        model_vectors = [torch.nn.utils.parameters_to_vector(model_dict.parameters()) for model_dict in model_dicts]

        euclidean_distances = [torch.dist(global_vector, mv, p=2).item() for mv in model_vectors]  
        cosine_distances = [1 - torch.cosine_similarity(global_vector, mv, dim=0).item() for mv in model_vectors]  

        #self.save_distances(euclidean_distances, cosine_distances)  

        clusters_cos, cluster_radius_cos = self.Clustering(cosine_distances)  
        clusters_euc, cluster_radius_euc = self.Clustering(euclidean_distances)  

        min_radius = min(cluster_radius_cos, cluster_radius_euc)  
        min_list = clusters_cos if min_radius == cluster_radius_cos else clusters_euc  

        S1_idxs = [idx for idx, val in enumerate(min_list) if val == 1]  
        #np.save(f'./matix/{self.args.dataset}_{self.args.attack_type}_iter{self.iters}_S1_idxs.npy', np.array(S1_idxs))  
        S1_updates = [model_dicts[i] for i in S1_idxs]  
        cluters_time = (time.perf_counter() - cluters_time_start)
        logger.info("clutering time: {}s".format(cluters_time))
        KL = []  
        cos_dis_outs = []  

        class_loaders = self.create_class_loaders()  
        LKM_time_start = time.perf_counter()
        for update in S1_updates:  
            KL_losses = []  
            for class_loader in class_loaders:  
                KL_loss = self.test(baseline, update, class_loader)  
                KL_losses.append(KL_loss)  
                #cos_dis_outs.append(cos_dis)  
            KL.append(KL_losses)  

        KL_t = torch.tensor(KL).cuda()  # This will have shape (num_updates, num_classes)  
        KL_dis = torch.norm(KL_t[:, None, :] - KL_t[None, :, :], dim=2, p=2).cpu().detach().numpy() 
        print(KL_t.shape, KL_dis.shape)
        #self.save_KL_data(KL_dis, KL_t)  
        # clustering = DBSCAN(eps=0.5, min_samples=2).fit(KL_dis)  
        # labels = clustering.labels_  
        
        # if len(set(labels)) < 2:  
            # Fallback to AgglomerativeClustering if DBSCAN fails  
        clustering = AgglomerativeClustering(n_clusters=2).fit(KL_dis)  
        #labels = clustering.labels_
        #clustering = AgglomerativeClustering(affinity="precomputed", linkage="average", n_clusters=2).fit(KL_dis)  
        flag = 1 if np.sum(clustering.labels_) > len(KL_t) // 2 else 0  
        S2_idxs = [idx for idx, label in enumerate(clustering.labels_) if label == flag]
        LKM_time=(time.perf_counter() - LKM_time_start)
        logger.info("LKM time: {}s".format(LKM_time))
        S2_updates = [S1_updates[i] for i in S2_idxs]
        print(len(S2_updates))
        #np.save(f'./matix/{self.args.dataset}_{self.args.attack_type}_iter{self.iters}_S2_idxs.npy', np.array(S2_idxs))  

        return S2_updates  

    def save_distances(self, euclidean_distances, cosine_distances):  
        np.save(f'./matix/{self.args.dataset}_{self.args.attack_type}_iter{self.iters}_euclidean_distances.npy', np.array(euclidean_distances))  
        np.save(f'./matix/{self.args.dataset}_{self.args.attack_type}_iter{self.iters}_cosine_distances.npy', np.array(cosine_distances))  

    def save_KL_data(self, KL_dis, KL_t):  
        np.save(f'./matix/{self.args.dataset}_{self.args.attack_type}_iter{self.iters}_KL_dis.npy', KL_dis)  
        np.save(f'./matix/{self.args.dataset}_{self.args.attack_type}_iter{self.iters}_KL_t.npy', KL_t.cpu().detach().numpy())  

    def create_class_loaders(self):  
        targets = torch.tensor(self.local_data.targets).clone().detach()
        num_classes = 10  # Assuming MNIST or CIFAR10  
        return [  
            DataLoader(  
                Subset(self.local_data, (targets == class_idx).nonzero().squeeze()),  
                batch_size=self.args.local_bs,  
                shuffle=False,  
                num_workers=4,  
                pin_memory=True  
            ) for class_idx in range(num_classes)  
        ]  

    @staticmethod  
    def average_weights(w, marks):  
        w_avg = copy.deepcopy(w[0])  
        for key in w_avg.keys():  
            w_avg[key] = sum(w[i][key] * marks[i] for i in range(len(w))) / sum(marks)  
        return w_avg
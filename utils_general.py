from utils_libs import *
from utils_dataset import *
from utils_models import *
# Global parameters
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter

import time
max_norm = 10
# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay = None):
    acc_overall = 0; loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    batch_size = min(6000, data_x.shape[0])
    batch_size = min(2000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval(); model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_overall += loss.item()

            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()            
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct
    
    
    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay/2 * np.sum(params * params)
        
    model.train()
    return loss_overall, acc_overall / n_tst

# --- Helper functions

def avg_models(mdl, clnt_models, weight_list):
    n_node = len(clnt_models)
    dict_list = list(range(n_node));
    for i in range(n_node):
        dict_list[i] = copy.deepcopy(dict(clnt_models[i].named_parameters()))

    param_0 = clnt_models[0].named_parameters()
    
    for name, param in param_0:
        param_ = weight_list[0] * param.data
        for i in list(range(1 ,n_node)):
            param_ = param_ + weight_list[i] * dict_list[i][name].data
        dict_list[0][name].data.copy_(param_)
        
    mdl.load_state_dict(dict_list[0])
    
    # Remove dict_list from memory
    del dict_list
    
    return mdl

def train_model(model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)                          
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train(); model = model.to(device)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)
    
    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f" 
              %(0, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
    else:
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" 
              %(0, acc_trn, loss_trn, scheduler.get_lr()[0]))
    
    model.train()
    
    for e in range(epoch):
        # Training
        
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):

            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            
            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients to prevent exploding
            optimizer.step()

        if (e+1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f" 
                      %(e+1, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" %(e+1, acc_trn, loss_trn, scheduler.get_lr()[0]))
    
            
            model.train()
        scheduler.step()
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_model_prox(model, cld_mdl_param, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)                          
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train(); model = model.to(device)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool) 
    
    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f" 
              %(0, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
    else:
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" 
              %(0, acc_trn, loss_trn, scheduler.get_lr()[0]))
    
    model.train()
    
    for e in range(epoch):
        # Training
        
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):

            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = 0.0001 * torch.sum((local_par_list-cld_mdl_param)* (local_par_list-cld_mdl_param))
            
            loss = loss / list(batch_y.size())[0]+loss_algo
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients to prevent exploding
            optimizer.step()

        if (e+1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f" 
                      %(e+1, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" %(e+1, acc_trn, loss_trn, scheduler.get_lr()[0]))
    
            
            model.train()
        scheduler.step()
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model





def train_scaffold_mdl(model, model_func, state_params_diff, trn_x, trn_y, 
                    learning_rate, batch_size, n_minibatch, print_per,
                    weight_decay, dataset_name, sch_step, sch_gamma):
    
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train(); model = model.to(device)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()
    
    n_par = get_mdl_params([model_func()]).shape[1]
    
    n_iter_per_epoch = int(np.ceil(n_trn/batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)
    
    count_step = 0
    is_done = False
    
    step_loss = 0; n_data_step = 0
    for e in range(epoch):
        # Training
        ###
        if is_done:
            break
        ###        
        
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            ####
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            ###
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = torch.sum(local_par_list * state_params_diff)
            
            loss = loss_f_i + loss_algo

            ###
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients to prevent exploding
            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0]; n_data_step += list(batch_y.size())[0]

            if (count_step) % print_per == 0:
                step_loss /= n_data_step
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    step_loss += (weight_decay)/2 * np.sum(params * params)

                print("Step %3d, Training Loss: %.4f, LR: %.5f"
                      %(count_step, step_loss, scheduler.get_lr()[0]))
                step_loss = 0; n_data_step = 0
            
            model.train()
        scheduler.step()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
        idx += length
    
    mdl.load_state_dict(dict_param)    
    return mdl


def get_mdl_params(model_list, n_par=None):
    
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))
    
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def train_model_alg(model, model_func, alpha_coef, avg_mdl_param, hist_params_diff, trn_x, trn_y, 
                    learning_rate, batch_size, epoch, print_per,
                    weight_decay, dataset_name, sch_step, sch_gamma):
    
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)

    model.train(); model = model.to(device)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()
    
    n_par = get_mdl_params([model_func()]).shape[1]
    
    
    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

                
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))
            
            loss = loss_f_i + loss_algo

            ###
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef+weight_decay)/2 * np.sum(params * params)
            
            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  %(e+1, epoch_loss, scheduler.get_lr()[0]))
            
            
            model.train()
        scheduler.step()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model






def train_model_FedDC(model, model_func, alpha, local_update_last, global_update_last, global_model_param, hist_i, trn_x, trn_y, 
                    learning_rate, batch_size, epoch, print_per,
                    weight_decay, dataset_name, sch_step, sch_gamma):
    
    n_trn = trn_x.shape[0]
    state_update_diff = torch.tensor(-local_update_last+ global_update_last,  dtype=torch.float32, device=device)  
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train(); model = model.to(device)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()
    
    n_par = get_mdl_params([model_func()]).shape[1]
    
    
    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

                
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            local_parameter = None
            for param in model.parameters():
                if not isinstance(local_parameter, torch.Tensor):
                # Initially nothing to concatenate
                    local_parameter = param.reshape(-1)
                else:
                    local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)
            
            loss_cp = alpha/2 * torch.sum((local_parameter - (global_model_param - hist_i))*(local_parameter - (global_model_param - hist_i)))
            loss_cg = torch.sum(local_parameter * state_update_diff) 
            

            loss = loss_f_i + loss_cp + loss_cg
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay)/2 * np.sum(params * params)
            
            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  %(e+1, epoch_loss, scheduler.get_lr()[0]))
            
            
            model.train()
        scheduler.step()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

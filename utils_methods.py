from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *
from tensorboardX import SummaryWriter

### Methods
def train_FedAvg(data_obj, act_prob ,learning_rate, batch_size, epoch, 
                                     com_amount, print_per, weight_decay, 
                                     model_func, init_model, sch_step, sch_gamma,
                                     save_period, suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedAvg_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)
   
    suffix += '_lrdecay%f' %lr_decay_per_round
    suffix += '_seed%d' %rand_seed

    n_clnt=data_obj.n_client

    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
        
    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' %(data_path, data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)); trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%s/%s' %(data_path, data_obj.name, suffix))
    
    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i
                
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' 
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model
                
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt' 
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr//save_period] = fed_model
                
                
                if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    trn_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    trn_perf_all[:i+1] = np.load('%sModel/%s/%s/%dcom_trn_perf_all.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    tst_perf_all[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_all.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))

    
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):         
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            
            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' 
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
            
            all_model = model_func().to(device)
            all_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt' 
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
          
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            
            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False
            
            
                clnt_models[clnt] = model_func().to(device)
                
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_models[clnt] = train_model(clnt_models[clnt], trn_x, trn_y, 
                                                tst_x, tst_y, 
                                                learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per,
                                                weight_decay, 
                                                data_obj.dataset, sch_step, sch_gamma)
                
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights
            
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
            all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, 
                                             avg_model, data_obj.dataset, 0)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             all_model, data_obj.dataset, 0)
            tst_perf_all[i] = [loss_tst, acc_tst]

            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, 
                                             all_model, data_obj.dataset, 0)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            
            writer.add_scalars('Loss/train_wd', 
                   {
                       'Sel clients':get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, weight_decay)[0],
                       'All clients':get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, weight_decay)[0]
                   }, i
                  )
            
            writer.add_scalars('Loss/train', 
                   {
                       'Sel clients':trn_perf_sel[i][0],
                       'All clients':trn_perf_all[i][0]
                   }, i
                  )
            
            writer.add_scalars('Accuracy/train', 
                   {
                       'Sel clients':trn_perf_sel[i][1],
                       'All clients':trn_perf_all[i][1]
                   }, i
                  )
            
            
            writer.add_scalars('Loss/test', 
                   {
                       'Sel clients':tst_perf_sel[i][0],
                       'All clients':tst_perf_all[i][0]
                   }, i
                  )
            
            writer.add_scalars('Accuracy/test', 
                   {
                       'Sel clients':tst_perf_sel[i][1],
                       'All clients':tst_perf_all[i][1]
                   }, i
                  )
                        
            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt' 
                               %(data_path, data_obj.name, suffix, (i+1)))                
                torch.save(all_model.state_dict(), '%sModel/%s/%s/%dcom_all.pt' 
                               %(data_path, data_obj.name, suffix, (i+1)))
                
                np.save('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), trn_perf_sel[:i+1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])
                
                np.save('%sModel/%s/%s/%dcom_trn_perf_all.npy' %(data_path, data_obj.name, suffix, (i+1)), trn_perf_all[:i+1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_all.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_all[:i+1])
                
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)
                
                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('%sModel/%s/%s/%dcom_trn_perf_all.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_all.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
                
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all

def train_FedProx(data_obj, act_prob ,learning_rate, batch_size, epoch, 
                                     com_amount, print_per, weight_decay, 
                                     model_func, init_model, sch_step, sch_gamma,
                                     save_period, mu, weight_uniform = False, suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedProx_final_mu0.0001_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)

    suffix += '_lrdecay%f' %lr_decay_per_round
    suffix += '_seed%d' %rand_seed
    suffix += '_mu%f_WU%s' %(mu, weight_uniform)

    n_clnt=data_obj.n_client

    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)
        
    # Average them based on number of datapoints (The one implemented)
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))
    
        
    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' %(data_path, data_obj.name, suffix))
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)); trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1
    
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%s/%s' %(data_path, data_obj.name, suffix))
    
    
    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' 
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model
                
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt' 
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr//save_period] = fed_model
               
                
                if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    trn_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    trn_perf_all[:i+1] = np.load('%sModel/%s/%s/%dcom_trn_perf_all.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    tst_perf_all[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_all.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))

    
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):         
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            
            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([all_model], n_par)[0]
        

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' 
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
            
            all_model = model_func().to(device)
            all_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt' 
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
            cld_mdl_param = get_mdl_params([all_model], n_par)[0]
        
          
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness

            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False
                # Add regulariser during training
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                #local train
                clnt_models[clnt] = train_model_prox(clnt_models[clnt],cld_mdl_param_tensor, trn_x, trn_y,
                                                tst_x, tst_y,
                                                learning_rate * (lr_decay_per_round ** i),
                                                batch_size, epoch, print_per,
                                                weight_decay, 
                                                data_obj.dataset, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
            all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))
            cld_mdl_param= np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0)
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, 
                                             avg_model, data_obj.dataset, 0)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             all_model, data_obj.dataset, 0)
            tst_perf_all[i] = [loss_tst, acc_tst]

            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, 
                                             all_model, data_obj.dataset, 0)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            
            writer.add_scalars('Loss/train_wd', 
                   {
                       'Sel clients':get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, weight_decay)[0],
                       'All clients':get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, weight_decay)[0]
                   }, i
                  )
            
            writer.add_scalars('Loss/train', 
                   {
                       'Sel clients':trn_perf_sel[i][0],
                       'All clients':trn_perf_all[i][0]
                   }, i
                  )
            
            writer.add_scalars('Accuracy/train', 
                   {
                       'Sel clients':trn_perf_sel[i][1],
                       'All clients':trn_perf_all[i][1]
                   }, i
                  )
            
            
            writer.add_scalars('Loss/test', 
                   {
                       'Sel clients':tst_perf_sel[i][0],
                       'All clients':tst_perf_all[i][0]
                   }, i
                  )
            
            writer.add_scalars('Accuracy/test', 
                   {
                       'Sel clients':tst_perf_sel[i][1],
                       'All clients':tst_perf_all[i][1]
                   }, i
                  )            

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt' 
                               %(data_path, data_obj.name, suffix, (i+1)))                
                torch.save(all_model.state_dict(), '%sModel/%s/%s/%dcom_all.pt' 
                               %(data_path, data_obj.name, suffix, (i+1)))
                
                np.save('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), trn_perf_sel[:i+1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])
                
                np.save('%sModel/%s/%s/%dcom_trn_perf_all.npy' %(data_path, data_obj.name, suffix, (i+1)), trn_perf_all[:i+1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_all.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_all[:i+1])
                
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)
                
                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('%sModel/%s/%s/%dcom_trn_perf_all.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_all.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
                
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all

def train_SCAFFOLD(data_obj, act_prob ,learning_rate, batch_size, n_minibatch, 
                                     com_amount, print_per, weight_decay, 
                                     model_func, init_model, sch_step, sch_gamma,
                                     save_period, suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1, global_learning_rate=1):
    suffix = 'Scaffold_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_K%d_W%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, n_minibatch, weight_decay)
   
    suffix += '_lrdecay%f' %lr_decay_per_round
    suffix += '_seed%d' %rand_seed

    n_clnt=data_obj.n_client

    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt # normalize it
        
    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' %(data_path, data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)); trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); tst_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])
    state_params_diffs = np.zeros((n_clnt+1, n_par)).astype('float32') #including cloud state
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%s/%s' %(data_path, data_obj.name, suffix))
    
    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' 
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model
                
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt' 
                               %(data_path, data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr//save_period] = fed_model
                
                if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1))):
                    trn_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    trn_perf_all[:i+1] = np.load('%sModel/%s/%s/%dcom_trn_perf_all.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    
                    tst_perf_sel[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    tst_perf_all[:i+1] = np.load('%sModel/%s/%s/%dcom_tst_perf_all.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))                    # Get state_params_diffs
                    state_params_diffs = np.load('%sModel/%s/%s/%d_state_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1))
    if (trial) or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' %(data_path, data_obj.name, suffix, com_amount))):         
        clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            
            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' 
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
            
            all_model = model_func().to(device)
            all_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt' 
                       %(data_path, data_obj.name, suffix, (saved_itr+1))))
          
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            
            del clnt_models
            
            clnt_models = list(range(n_clnt))
            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0]
            
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False
            
            
                clnt_models[clnt] = model_func().to(device)
                
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                    
                # Scale down c
                state_params_diff_curr = torch.tensor(-state_params_diffs[clnt] + state_params_diffs[-1]/weight_list[clnt], dtype=torch.float32, device=device)
                
                clnt_models[clnt] = train_scaffold_mdl(clnt_models[clnt], model_func, state_params_diff_curr, trn_x, trn_y, 
                    learning_rate * (lr_decay_per_round ** i), batch_size, n_minibatch, print_per,
                    weight_decay, data_obj.dataset, sch_step, sch_gamma)
                
                curr_model_param = get_mdl_params([clnt_models[clnt]], n_par)[0]
                new_c = state_params_diffs[clnt] - state_params_diffs[-1] + 1/n_minibatch/learning_rate * (prev_params - curr_model_param)
                # Scale up delta c
                delta_c_sum += (new_c - state_params_diffs[clnt])*weight_list[clnt]
                state_params_diffs[clnt] = new_c
                
                clnt_params_list[clnt] = curr_model_param
                
            
            avg_model_params = global_learning_rate*np.mean(clnt_params_list[selected_clnts], axis = 0) + (1-global_learning_rate)*prev_params
           
            avg_model = set_client_from_params(model_func().to(device), avg_model_params) 
            
            state_params_diffs[-1] += 1 / n_clnt * delta_c_sum
            
            all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis = 0))
            
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, 
                                             avg_model, data_obj.dataset, 0)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            
            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             all_model, data_obj.dataset, 0)
            tst_perf_all[i] = [loss_tst, acc_tst]

            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, 
                                             all_model, data_obj.dataset, 0)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            
            
            writer.add_scalars('Loss/train_wd', 
                   {
                       'Sel clients':get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, weight_decay)[0],
                       'All clients':get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, weight_decay)[0]
                   }, i
                  )
            
            writer.add_scalars('Loss/train', 
                   {
                       'Sel clients':trn_perf_sel[i][0],
                       'All clients':trn_perf_all[i][0]
                   }, i
                  )
            
            writer.add_scalars('Accuracy/train', 
                   {
                       'Sel clients':trn_perf_sel[i][1],
                       'All clients':trn_perf_all[i][1]
                   }, i
                  )
            
            
            writer.add_scalars('Loss/test', 
                   {
                       'Sel clients':tst_perf_sel[i][0],
                       'All clients':tst_perf_all[i][0]
                   }, i
                  )
            
            writer.add_scalars('Accuracy/test', 
                   {
                       'Sel clients':tst_perf_sel[i][1],
                       'All clients':tst_perf_all[i][1]
                   }, i
                  )
            
            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt' 
                               %(data_path, data_obj.name, suffix, (i+1)))                
                torch.save(all_model.state_dict(), '%sModel/%s/%s/%dcom_all.pt' 
                               %(data_path, data_obj.name, suffix, (i+1)))
                
                np.save('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), trn_perf_sel[:i+1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])
                
                np.save('%sModel/%s/%s/%dcom_trn_perf_all.npy' %(data_path, data_obj.name, suffix, (i+1)), trn_perf_all[:i+1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_all.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_perf_all[:i+1])
                
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)
                # save state_params_diffs
                np.save('%sModel/%s/%s/%d_state_params_diffs.npy' %(data_path, data_obj.name, suffix, (i+1)), state_params_diffs)
                
                
                if (i+1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period)):
                        os.remove('%sModel/%s/%s/%dcom_trn_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('%sModel/%s/%s/%dcom_trn_perf_all.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_all.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                        os.remove('%sModel/%s/%s/%d_state_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1-save_period)) 
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
                
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all


def train_FedDyn(data_obj, act_prob,
                  learning_rate, batch_size, epoch, com_amount, print_per, 
                  weight_decay,  model_func, init_model, alpha_coef,
                  sch_step, sch_gamma, save_period,
                  suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix  = 'FedDyn_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' %(save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' %rand_seed
    suffix += '_lrdecay%f' %lr_decay_per_round
    
    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' %(data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' %(data_path, data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    avg_ins_mdls = list(range(n_save_instances)) # Avg active clients
    avg_all_mdls = list(range(n_save_instances)) # Avg all clients
    avg_cld_mdls = list(range(n_save_instances)) # Cloud models 
    
    trn_sel_clt_perf = np.zeros((com_amount, 2))
    tst_sel_clt_perf = np.zeros((com_amount, 2))
    
    trn_all_clt_perf = np.zeros((com_amount, 2))
    tst_all_clt_perf = np.zeros((com_amount, 2))
    
    trn_cur_cld_perf = np.zeros((com_amount, 2))
    tst_cur_cld_perf = np.zeros((com_amount, 2))
    
    n_par = len(get_mdl_params([model_func()])[0])
    
    hist_params_diffs = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list  = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1
            
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%s/%s' %(data_path, data_obj.name, suffix))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/ins_avg_%dcom.pt' 
                               %(data_path, data_obj.name, suffix, i+1)):
                saved_itr = i
                
                ####
                fed_ins = model_func()
                fed_ins.load_state_dict(torch.load('%sModel/%s/%s/ins_avg_%dcom.pt' %(data_path, data_obj.name, suffix, i+1)))
                fed_ins.eval()
                fed_ins = fed_ins.to(device)
                
                # Freeze model
                for params in fed_ins.parameters():
                    params.requires_grad = False

                avg_ins_mdls[saved_itr//save_period] = fed_ins 
                
                ####
                fed_all = model_func()
                fed_all.load_state_dict(torch.load('%sModel/%s/%s/all_avg_%dcom.pt' %(data_path, data_obj.name, suffix, i+1)))
                fed_all.eval()
                fed_all = fed_all.to(device)
                
                # Freeze model
                for params in fed_all.parameters():
                    params.requires_grad = False

                avg_all_mdls[saved_itr//save_period] = fed_all 
                
                
                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' %(data_path, data_obj.name, suffix, i+1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr//save_period] = fed_cld 
                
                if os.path.exists('%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' %(data_path, data_obj.name, suffix, (i+1))):

                    trn_sel_clt_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    tst_sel_clt_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_tst_sel_clt_perf.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    trn_all_clt_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_trn_all_clt_perf.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    tst_all_clt_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_tst_all_clt_perf.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    trn_cur_cld_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, (i+1)))
                    tst_cur_cld_perf[:i+1] = np.load('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, (i+1)))

                    # Get hist_params_diffs
                    hist_params_diffs = np.load('%sModel/%s/%s/%d_hist_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1))
                    clnt_params_list = np.load('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1))
                    
                                   
                      
    if (trial) or (not os.path.exists('%sModel/%s/%s/ins_avg_%dcom.pt' %(data_path, data_obj.name, suffix, com_amount))):
        
        
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            
            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
        
        else:            
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
        
    
        for i in range(saved_itr+1, com_amount):
            # Train if doesn't exist    
            ### Fix randomness
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
            
            del clnt_models
            clnt_models = list(range(n_clnt))
            
            for clnt in selected_clnts:
                # Train locally 
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                
                clnt_models[clnt] = model_func().to(device)
                
                model = clnt_models[clnt]
                # Warm start from current avg model
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                # Scale down
                alpha_coef_adpt = alpha_coef / weight_list[clnt] # adaptive alpha coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[clnt], dtype=torch.float32, device=device)
                clnt_models[clnt] = train_model_alg(model, model_func, alpha_coef_adpt, 
                                                     cld_mdl_param_tensor, hist_params_diffs_curr, 
                                                     trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), 
                                                     batch_size, epoch, print_per, weight_decay, 
                                                     data_obj.dataset, sch_step, sch_gamma)
                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                hist_params_diffs[clnt] += curr_model_par-cld_mdl_param
                clnt_params_list[clnt] = curr_model_par
                        
            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis = 0)
            
            cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)
            
            avg_model_sel = set_client_from_params(model_func(), avg_mdl_param_sel)
            all_model     = set_client_from_params(model_func(), np.mean(clnt_params_list, axis = 0))
            
            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param) 
            
            #####

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, 
                                             avg_model_sel, data_obj.dataset, 0)
            print("**** Cur Sel Communication %3d, Cent Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            trn_sel_clt_perf[i] = [loss_tst, acc_tst]
            
            #####

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, 
                                             all_model, data_obj.dataset, 0)
            print("**** Cur All Communication %3d, Cent Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            trn_all_clt_perf[i] = [loss_tst, acc_tst]
            
            #####

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, 
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Cent Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            trn_cur_cld_perf[i] = [loss_tst, acc_tst]
            
                        
            writer.add_scalars('Loss/train', 
                   {
                       'Sel clients':trn_sel_clt_perf[i][0],
                       'All clients':trn_all_clt_perf[i][0],
                       'Current cloud':trn_cur_cld_perf[i][0]
                   }, i
                  )
            
            writer.add_scalars('Accuracy/train', 
                   {
                       'Sel clients':trn_sel_clt_perf[i][1],
                       'All clients':trn_all_clt_perf[i][1],
                       'Current cloud':trn_cur_cld_perf[i][1]
                   }, i
                  )
            
            writer.add_scalars('Loss/train_wd', 
                   {
                       'Sel clients':get_acc_loss(cent_x, cent_y, avg_model_sel, data_obj.dataset, weight_decay)[0],
                       'All clients':get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, weight_decay)[0],
                       'Current cloud':get_acc_loss(cent_x, cent_y, cur_cld_model, data_obj.dataset, weight_decay)[0]
                   }, i
                  )
            
            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             avg_model_sel, data_obj.dataset, 0)
            print("**** Cur Sel Communication %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            tst_sel_clt_perf[i] = [loss_tst, acc_tst]
            
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             all_model, data_obj.dataset, 0)
            print("**** Cur All Communication %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            tst_all_clt_perf[i] = [loss_tst, acc_tst]
            
            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, 
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f" 
                  %(i+1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]
            
            
            writer.add_scalars('Loss/test', 
                   {
                       'Sel clients':tst_sel_clt_perf[i][0],
                       'All clients':tst_all_clt_perf[i][0],
                       'Current cloud':tst_cur_cld_perf[i][0]
                   }, i
                  )
            
            writer.add_scalars('Accuracy/test', 
                   {
                       'Sel clients':tst_sel_clt_perf[i][1],
                       'All clients':tst_all_clt_perf[i][1],
                       'Current cloud':tst_cur_cld_perf[i][1]
                   }, i
                  )     
            
            if (not trial) and ((i+1) % save_period == 0):
                torch.save(avg_model_sel.state_dict(), '%sModel/%s/%s/ins_avg_%dcom.pt' 
                           %(data_path, data_obj.name, suffix, (i+1)))
                torch.save(all_model.state_dict(), '%sModel/%s/%s/all_avg_%dcom.pt' 
                           %(data_path, data_obj.name, suffix, (i+1)))
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt' 
                           %(data_path, data_obj.name, suffix, (i+1)))
                
                
                np.save('%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' %(data_path, data_obj.name, suffix, (i+1)), trn_sel_clt_perf[:i+1])
                np.save('%sModel/%s/%s/%d_com_tst_sel_clt_perf.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_sel_clt_perf[:i+1])
                np.save('%sModel/%s/%s/%d_com_trn_all_clt_perf.npy' %(data_path, data_obj.name, suffix, (i+1)), trn_all_clt_perf[:i+1])
                np.save('%sModel/%s/%s/%d_com_tst_all_clt_perf.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_all_clt_perf[:i+1])

                np.save('%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, (i+1)), trn_cur_cld_perf[:i+1])
                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, (i+1)), tst_cur_cld_perf[:i+1])
                    
                # save hist_params_diffs

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' %(data_path, data_obj.name, suffix, (i+1)), hist_params_diffs)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, (i+1)), clnt_params_list)
                     

                if (i+1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_com_tst_sel_clt_perf.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_com_trn_all_clt_perf.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_com_tst_all_clt_perf.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1-save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' %(data_path, data_obj.name, suffix, i+1-save_period))

                    
                    
            if ((i+1) % save_period == 0):
                avg_ins_mdls[i//save_period] = avg_model_sel
                avg_all_mdls[i//save_period] = all_model
                avg_cld_mdls[i//save_period] = cur_cld_model
                    
    return avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf, trn_all_clt_perf, tst_all_clt_perf
    
from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *
from tensorboardX import SummaryWriter
### Methods



def train_FedDC(data_obj, act_prob,n_minibatch,
                  learning_rate, batch_size, epoch, com_amount, print_per, 
                  weight_decay,  model_func, init_model, alpha_coef,
                  sch_step, sch_gamma, save_period,
                  suffix = '', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix  = 'FedDC_' + str(alpha_coef)+suffix
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
    avg_ins_mdls = list(range(n_save_instances)) 
    avg_all_mdls = list(range(n_save_instances)) 
    avg_cld_mdls = list(range(n_save_instances)) 

    trn_sel_clt_perf = np.zeros((com_amount, 2))
    tst_sel_clt_perf = np.zeros((com_amount, 2))
    
    trn_all_clt_perf = np.zeros((com_amount, 2))
    tst_all_clt_perf = np.zeros((com_amount, 2))
    
    trn_cur_cld_perf = np.zeros((com_amount, 2))
    tst_cur_cld_perf = np.zeros((com_amount, 2))
    
    n_par = len(get_mdl_params([model_func()])[0])

    parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0]
    clnt_params_list  = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    ###
    state_gadient_diffs = np.zeros((n_clnt+1, n_par)).astype('float32') #including cloud state
    

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
                    parameter_drifts = np.load('%sModel/%s/%s/%d_hist_params_diffs.npy' %(data_path, data_obj.name, suffix, i+1))
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
            inc_seed = 0
            while(True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break

            global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device) #Theta
            del clnt_models
            clnt_models = list(range(n_clnt))
            delta_g_sum = np.zeros(n_par)
            
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                local_update_last = state_gadient_diffs[clnt] # delta theta_i
                global_update_last = state_gadient_diffs[-1]/weight_list[clnt] #delta theta
                alpha = alpha_coef / weight_list[clnt] 
                hist_i = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device) #h_i
                clnt_models[clnt] = train_model_FedDC(model, model_func, alpha,local_update_last, global_update_last,global_mdl, hist_i, 
                                                     trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), 
                                                     batch_size, epoch, print_per, weight_decay, data_obj.dataset, sch_step, sch_gamma)


                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                delta_param_curr = curr_model_par-cld_mdl_param
                parameter_drifts[clnt] += delta_param_curr 
                beta = 1/n_minibatch/learning_rate
                
                state_g = local_update_last - global_update_last + beta * (-delta_param_curr) 
                delta_g_cur = (state_g - state_gadient_diffs[clnt])*weight_list[clnt] 
                delta_g_sum += delta_g_cur
                state_gadient_diffs[clnt] = state_g 
                clnt_params_list[clnt] = curr_model_par 
                

                        
            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis = 0)
            delta_g_cur = 1 / n_clnt * delta_g_sum  
            state_gadient_diffs[-1] += delta_g_cur  
            
            cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)
            
            avg_model_sel = set_client_from_params(model_func(), avg_mdl_param_sel)
            all_model     = set_client_from_params(model_func(), np.mean(clnt_params_list, axis = 0))
            
            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param) 

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
                    
                # save parameter_drifts

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' %(data_path, data_obj.name, suffix, (i+1)), parameter_drifts)
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



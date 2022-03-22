from utils_general import *
from utils_methods import *
from utils_methods_FedDC import train_FedDC
# Dataset initialization
data_path = 'Folder/' # The folder to save Data & Model
###
alpha=0.0
beta=0.0
iid_sol=True
iid_data=True
name_prefix="syn_alpha-"+str(alpha)+"_beta-"+str(beta)

n_dim = 30
n_clnt= 20
n_cls = 5
avg_data = 200

data_obj = DatasetSynthetic(alpha=alpha, beta=beta, iid_sol=iid_sol, iid_data=iid_data, n_dim=n_dim, n_clnt=n_clnt, n_cls=n_cls, avg_data=avg_data, data_path=data_path, name_prefix=name_prefix)



###
# Common hyperparameters
com_amount = 300
save_period = 100
weight_decay = 1e-5
batch_size = 10
act_prob= 0.15
model_name = 'Linear' # Model type
suffix = model_name
lr_decay_per_round = 1

# Model function
model_func = lambda : client_model(model_name, [n_dim, n_cls])
init_model = model_func()

# Initalise the model for all methods
with torch.no_grad():
    init_model.fc.weight = torch.nn.parameter.Parameter(torch.zeros(n_cls,n_dim))
    init_model.fc.bias = torch.nn.parameter.Parameter(torch.zeros(n_cls))

print('FedDC')

epoch = 10
alpha_coef = 0.005
learning_rate = 0.1

n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_clnt
n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)

n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
learning_rate = 0.1
print_per = 5



[avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf, trn_all_clt_perf, tst_all_clt_perf] = train_FedDC(data_obj=data_obj, act_prob=act_prob, n_minibatch=n_minibatch, 
                                    learning_rate=learning_rate, batch_size=batch_size, epoch=epoch, 
                                    com_amount=com_amount, print_per=print_per, weight_decay=weight_decay, 
                                    model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
                                    sch_step=1, sch_gamma=1,save_period=save_period, suffix=suffix, trial=False, 
                                    data_path=data_path, lr_decay_per_round=lr_decay_per_round)
#exit(0)
###
            

# Baselines    
####
print('FedDyn')

epoch = 10
alpha_coef = 1e-2
learning_rate = 0.1
print_per = epoch // 2

    
[avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf, trn_all_clt_perf, tst_all_clt_perf] = train_FedDyn(data_obj=data_obj, act_prob=act_prob,
                                    learning_rate=learning_rate, batch_size=batch_size, epoch=epoch, 
                                    com_amount=com_amount, print_per=print_per, weight_decay=weight_decay, 
                                    model_func=model_func, init_model=init_model, alpha_coef=alpha_coef,
                                    sch_step=1, sch_gamma=1,save_period=save_period, suffix=suffix, trial=False,
                                    data_path=data_path, lr_decay_per_round=lr_decay_per_round)

###
print('SCAFFOLD')

epoch = 10

n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_clnt
n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)

n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
learning_rate = 0.1
print_per = 5


[fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all] = train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob ,
                                    learning_rate=learning_rate, batch_size=batch_size, n_minibatch=n_minibatch, 
                                    com_amount=com_amount, print_per=n_minibatch//2, weight_decay=weight_decay, 
                                    model_func=model_func, init_model=init_model,
                                    sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix, 
                                    trial=False, data_path=data_path, lr_decay_per_round=lr_decay_per_round)

####
print('FedAvg')

epoch = 10
learning_rate = 0.1
print_per = 5

[fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all] = train_FedAvg(data_obj=data_obj, act_prob=act_prob ,
                                    learning_rate=learning_rate, batch_size=batch_size, epoch=epoch, 
                                    com_amount=com_amount, print_per=print_per, weight_decay=weight_decay, 
                                    model_func=model_func, init_model=init_model,
                                    sch_step=1, sch_gamma=1, save_period=save_period, suffix=suffix, 
                                    trial=False, data_path=data_path, lr_decay_per_round=lr_decay_per_round)
        
#### 
print('FedProx')

epoch = 10
learning_rate = 0.1
print_per = 5
mu = 1e-4


[fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all]  = train_FedProx(data_obj=data_obj, act_prob=act_prob ,
                                learning_rate=learning_rate, batch_size=batch_size, epoch=epoch, 
                                com_amount=com_amount, print_per=print_per, weight_decay=weight_decay, 
                                model_func=model_func, init_model=init_model, sch_step=1, sch_gamma=1,
                                save_period=save_period, mu=mu, suffix=suffix, trial=False,
                                data_path=data_path, lr_decay_per_round=lr_decay_per_round)
import torch
from client import *
from .server import Server
import numpy as np

class FedPDHyper(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):   
        super(FedPDHyper, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        self.gamma = 1.0
        self.gammas = torch.ones(args.total_client, dtype= torch.float32)
        self.last_global_update = torch.zeros(self.server_model_params_list.shape)
        self.h_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print(" Dual Variable Param List  --->  {:d} * {:d}".format(
                self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))
        
        # rebuild
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Local_dual_correction': torch.zeros((init_par_list.shape[0])),
        }
        self.Client = fedpdhyper

    def _select_clients_(self, t):
        # select active clients ID
        inc_seed = 0
        while(True):
            np.random.seed(t + self.args.seed + inc_seed)
            act_list = np.random.uniform(size=self.args.total_client)
            act_clients = act_list <= self.args.active_ratio
            selected_clients = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clients) != 0:
                return selected_clients
    
    
    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta\
                                    * (self.server_model_params_list - self.clients_params_list[client]))
        
        # combination of dual variable and global server model
        # local gradient is g - hi + alpha(wk - wt)
        #                  ---       --------
        #                grad      weight-decay
        # Therefore, -hi - alpha*wt are communicated as Local_dual_correction term
        # self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.server_model_params_list)
        self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.comm_vecs['Params_list'])
        self.last_global_update = Averaged_update.clone().detach()

    
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # FedDyn (ServerOpt)
        # w(t+1) = average_s[wi(t)] + average_c[h(t)]
        delta_global = (self.last_global_update * Averaged_update).sum()
        print("The global update is {%.3f}" % (delta_global).item())
        self.gamma = torch.clamp(self.gamma + delta_global, min=1.0, max=5.0)
        print("The global gamma is {%.3f}" % (self.gamma).item())
        return Averaged_model + torch.mean(self.h_params_list, dim=0)
    
    
    def postprocess(self, client,received_vecs):
        delta = (self.clients_updated_params_list[client] * self.last_global_update).sum()
        self.gammas[client] = self.gamma # global -> local  
        self.gammas[client] = torch.clamp(self.gammas[client] + delta, min=1.0, max=5.0)
        print("The client gamma of {%d} is {%.3f}" % (client, self.gammas[client].item()))
        self.h_params_list[client] += self.gammas[client] * self.clients_updated_params_list[client]
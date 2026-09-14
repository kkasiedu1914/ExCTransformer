import time
import random
import time
import sys
import threading
from copy import deepcopy
from itertools import combinations

import networkx as nx
import numpy as np
import scipy.special
import torch
from tqdm import tqdm
from sklearn.linear_model import (Lasso,
                                  LinearRegression, Ridge)
from sklearn.metrics import r2_score


class GraphSVX_explainer():

  """ 
  GraphSVX- Shapley Explanations for graphs, adapted from: https://github.com/AlexDuvalinho/GraphSVX/
  """
  
  def __init__(self, data, model, gpu=False):
    self.model = model
    self.data = data   # Graph to be explained should be a torch_geometric.data.Data object
    self.gpu = gpu
    self.neighbours = None  #  nodes considered
    self.F = None  # number of features considered
    self.M = None  # number of features and nodes considered
    self.base_values = []
    self.surrogate_metrics = None
    self._last_surrogate_metrics = None
    self.progress_log_every_s = None
    self.coalition_log_every_s = None
    self.all_coalition_cap = 20000
    self.last_requested_num_samples = None
    self.last_effective_num_samples = None

  def explain_crystal(self,                 #Removed multiclass parameter,  hops
                       graph_indices=[0],
                       num_samples=10,
                       info=True,
                       fullempty=None,
                       S=3, 
                       include_additional_tokens=False,
                       use_mask_only=False,
                       args_hv='auto',
                       args_feat='Expectation',
                       args_coal='Smarter',
                       args_g='WLS',
                       regu=None,
                       seed=None,
                       ):
        """ Explains prediction for a graph classification task - GraphSVX method

        params:
            node_indexes (list, optional): indexes of the nodes of interest. Defaults to [0].
            hops (int, optional): number k of k-hop neighbours to consider in the subgraph 
                                                    around node_index. Defaults to 2.
            num_samples (int, optional): number of coalitions sampled and used to approx shapley values. 
                                                    Defaults to 10.
            info (bool, optional): Print information about explainer's inner workings. 
                                                    And include vizualisation. Defaults to True.
            multiclass (bool, optional): extension - consider predicted class only or all classes
            fullempty (bool, optional): enforce high weight for full and empty coalitions
            S (int, optional): maximum size of coalitions that are favoured in mask generation phase 
            args_hv (str, optional): strategy used to convert simplified input z to original
                                                    input space z'. Use "auto" to select
                                                    classification vs regression from the model.
            args_feat (str, optional): way to switch off and discard node features (0 or expectation)
            args_coal (str, optional): how we sample coalitions z
            args_g (str, optional): method used to train model g on (z, f(z'))
            regu (int, optional): extension - apply regularisation to balance importance granted
                                                    to nodes vs features
            include_additional_tokens (bool, optional): when True, appended non-atom tokens
                                                    (e.g., void-fraction) are treated as players.
            seed (int | None, optional): RNG seed for reproducible coalition sampling. When None,
                                                    sampling remains stochastic.
            vizu (bool, optional): creates vizualisation or not

        Returns:
                [tensors]: shapley values for features/neighbours that influence node v's pred
                        and base value
        """

        # Time
        start = time.time()

        # --- Explain several nodes iteratively ---
        self.phi_list = []
        self.base_values = []
        self.surrogate_metrics = []
        if len(graph_indices) > 1 and not self.gpu:
            print(
                "Warning: batching GraphSVX explanations on CPU can be very slow. "
                "Consider enabling GPU for batch runs."
            )
        for graph_index in graph_indices: #This was desinged for a batch of graphs (make adjustments in future updates)
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if self.gpu and torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            # Compute true prediction for original instance via explained GNN model
            device = torch.device(
                'cuda' if self.gpu and torch.cuda.is_available() else 'cpu'
            )
            self.model = self.model.to(device)
            if torch.is_tensor(self.data.get("x")):
                self.data["x"] = self.data["x"].to(device)
            with torch.no_grad():
                pred = self.model.infer_out(
                    self.data,
                    self.data["pred_index"],
                    configs=self.data["config"],
                )
       

            # Only treat atoms as players (exclude appended added_feats tokens).
            num_atoms = int(self.data.get("num_atoms", self.data["x"].shape[0]))
            num_add_tokens = int(self.data.get("num_add_tokens", 0))
            self.neighbours = list(range(num_atoms))
            if include_additional_tokens and num_add_tokens > 0:
                # Treat additional feature tokens (e.g., void fraction) as explicit players.
                self.neighbours.extend(range(num_atoms, num_atoms + num_add_tokens))
            D = len(self.neighbours)

            self.F = 0
            self.M = self.F+D
            
            # Def range of endcases considered
            args_K = S

            # --- MASK GENERATOR ---
            if info:
                print(
                    f"Coalition build started | num_samples={num_samples} S={S} players={D}",
                    flush=True,
                )
            coalition_event = None
            coalition_thread = None
            if info and self.coalition_log_every_s:
                interval = self.coalition_log_every_s
                coalition_event = threading.Event()

                def _coalition_logger():
                    while not coalition_event.wait(interval):
                        print("Still building coalitions...", flush=True)

                coalition_thread = threading.Thread(
                    target=_coalition_logger, daemon=True
                )
                coalition_thread.start()

            z_, weights = self.mask_generation(num_samples, args_coal, args_K, D, regu) #removed info parameter, the weights here are not the shap values the phi 
            # below rather are 
            # Stash for optional surrogate comparisons.
            effective_num_samples = int(z_.shape[0])
            self.last_requested_num_samples = int(num_samples)
            self.last_effective_num_samples = effective_num_samples
            self.z_ = z_
            self.weights = weights
            if info and effective_num_samples != int(num_samples):
                print(
                    "Coalition sample count adjusted | "
                    f"requested={num_samples} effective={effective_num_samples} "
                    f"args_coal={args_coal}",
                    flush=True,
                )
            if coalition_event is not None:
                coalition_event.set()
            if coalition_thread is not None:
                coalition_thread.join(timeout=0.1)

            
            # Discard full and empty coalition if specified
            if fullempty:
                weights[(weights == 1000).nonzero()] = 0

            # --- GRAPH GENERATOR ---
            # Create dataset (z, f(GEN(z'))), stored as (z_, fz)
            # Retrieve z' from z and x_v, then compute f(z')
            if info:
                print("Explanation started", flush=True)

            if args_hv is None or str(args_hv).lower() == "auto":
                task_types = getattr(self.model, "current_tasks", []) or []
                if "classification" in task_types and "regression" not in task_types:
                    args_hv = "graph_classification"
                else:
                    args_hv = "graph_regression"

            if args_hv == "graph_regression":
                fz = self.graph_regression(
                    graph_index,
                    effective_num_samples,
                    D,
                    z_,
                    args_K,
                    args_feat,
                    pred,
                    use_mask_only=use_mask_only,
                )   # self, graph_index, num_samples, D, z_, args_K, args_feat, true_pred
            elif args_hv == "graph_classification":
                fz = self.graph_classification(
                    graph_index,
                    effective_num_samples,
                    D,
                    z_,
                    args_K,
                    args_feat,
                    pred,
                    use_mask_only=use_mask_only,
                )
            else:
                raise ValueError(f"Unknown args_hv: {args_hv}")
            if fz.shape[0] != effective_num_samples or weights.shape[0] != effective_num_samples:
                raise ValueError(
                    "Coalition/sample length mismatch after mask generation: "
                    f"requested={num_samples}, effective={effective_num_samples}, "
                    f"weights={weights.shape[0]}, fz={fz.shape[0]}, args_coal={args_coal}"
                )
            self.fz = fz
            

            # --- EXPLANATION GENERATOR --- 
            # Train Surrogate Weighted Linear Regression - learns shapley values
            phi, base_value = eval('self.' + args_g)(z_, weights,fz, info)

            self.phi_list.append(phi)
            self.base_values.append(base_value)
            self.surrogate_metrics.append(self._last_surrogate_metrics)
  
        phi_arr = np.asarray(self.phi_list)
        base_arr = np.asarray(self.base_values)
        return torch.from_numpy(phi_arr), torch.from_numpy(base_arr)

  def _record_surrogate_metrics(self, name, y_true, y_pred, weights):
        metrics = {
            "model": name,
            "r2": float(r2_score(y_true, y_pred)),
            "weighted_r2": float(r2_score(y_true, y_pred, sample_weight=weights)),
            "n_samples": int(len(y_true)),
        }
        self._last_surrogate_metrics = metrics
        return metrics

  def feature_selection(self, node_index, args_feat):
        """ Select features who truly impact prediction
        Others will receive a 0 shapley value anyway 

        Params:
            node_index (int): node index
            args_feat (str): strategy utilised to select 
                                important featutres

        Returns:
            [tensor]: list of important features' index
            [tensor]: list of discarded features' index
        """
        
        #Only consider relevant features in explanations
        discarded_feat_idx = []
        if args_feat == 'All':
            # Select all features 
            self.F = self.data.x[node_index, :].shape[0]
            feat_idx = torch.unsqueeze(
                torch.arange(self.data.num_nodes), 1)
        elif args_feat == 'Null':
            # Select features whose value is non-null
            feat_idx = self.data.x[node_index, :].nonzero()
            self.F = feat_idx.size()[0]
        else:
            # Select features whose value is different from dataset mean value
            std = self.data.x.std(axis=0)
            mean = self.data.x.mean(axis=0)
            mean_subgraph = self.data.x[node_index, :]
            mean_subgraph = torch.where(mean_subgraph >= mean - 0.25*std, mean_subgraph,
                                        torch.ones_like(mean_subgraph)*100)
            mean_subgraph = torch.where(mean_subgraph <= mean + 0.25*std, mean_subgraph,
                                        torch.ones_like(mean_subgraph)*100)
            feat_idx = (mean_subgraph == 100).nonzero()
            discarded_feat_idx = (mean_subgraph != 100).nonzero()
            self.F = feat_idx.shape[0]

        return feat_idx, discarded_feat_idx

  def mask_generation(self, num_samples, args_coal, args_K, D, regu):
            """ Applies selected mask generator strategy 

            Args:
                num_samples (int): number of samples for GraphSVX 
                args_coal (str): mask generator strategy 
                args_K (int): size param for indirect effect 
                D (int): number of nodes considered after selection
                info (bool): print information or not 
                regu (int): balances importance granted to nodes and features

            Returns:
                [tensor] (num_samples, M): dataset of samples/coalitions z' 
                [tensor] (num_samples): vector of kernel weights corresponding to samples 
            """
            if args_coal == 'SmarterSeparate' or args_coal == 'NewSmarterSeparate':
                weights = torch.zeros(num_samples, dtype=torch.float64)
                if self.F==0 or D==0:
                    num = int(num_samples * self.F/self.M)
                elif regu != None:
                    num = int(num_samples * regu)
                    #num = int( num_samples * ( self.F/self.M + ((regu - 0.5)/0.5)  * (self.F/self.M) ) )    
                else: 
                    num = int(0.5* num_samples/2 + 0.5 * num_samples * self.F/self.M)
                # Features only
                z_bis = eval('self.' + args_coal)(num, args_K, 1)  
                z_bis = z_bis[torch.randperm(z_bis.size()[0])]
                s = (z_bis != 0).sum(dim=1)
                weights[:num] = self.shapley_kernel(s, self.F)
                z_ = torch.zeros(num_samples, self.M)
                z_[:num, :self.F] = z_bis
                # Node only
                z_bis = eval('self.' + args_coal)(
                    num_samples-num, args_K, 0)  
                z_bis = z_bis[torch.randperm(z_bis.size()[0])]
                s = (z_bis != 0).sum(dim=1)
                weights[num:] = self.shapley_kernel(s, D)
                z_[num:, :] = torch.ones(num_samples-num, self.M)
                z_[num:, self.F:] = z_bis

            else:
                # If we choose to sample all possible coalitions
                if args_coal == 'All':
                    all_cap = int(getattr(self, "all_coalition_cap", 20000))
                    num_samples = min(all_cap, 2**self.M)

                # Coalitions: sample num_samples binary vectors of dimension M
                z_ = eval('self.' + args_coal)(num_samples, args_K, regu)

                # Shuffle them 
                z_ = z_[torch.randperm(z_.size()[0])]

                # Compute |z| for each sample z: number of non-zero entries
                s = (z_ != 0).sum(dim=1)

                # GraphSVX Kernel: define weights associated with each sample 
                weights = self.shapley_kernel(s, self.M)
                
            return z_, weights


  def NewSmarterSeparate(self, num_samples, args_K, regu):
        """Default mask sampler
        Generates feature mask and node mask independently
        Favours masks with a high weight + smart space allocation algorithm

        Args:
            num_samples (int): number of masks desired 
            args_K (int): maximum size of masks favoured
            regu (binary): nodes or features 

        Returns:
            tensor: dataset of samples
        """
        if regu == None:
            z_ = self.Smarter(num_samples, args_K, regu)
            return z_

        # Favour features - special coalitions don't study node's effect
        elif regu > 0.5:
            M = self.F
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples//2, M) # case k = 0 
            i = 2
            k = 1
            P = num_samples * 9/10
            while i < P and k <= min(args_K, M-1): 

                # All coalitions of order k can be sampled 
                if i + 2 * scipy.special.comb(M,k) <= P:
                    L = list(combinations(range(M), k))
                    for c in L:
                        z_[i,c] = torch.zeros(k)
                        z_[i+1,c] = torch.ones(k)
                        i += 2
            
                # All coalitions of order k cannot be sampled
                else: 
                    weight = torch.ones(M)
                    L = list(combinations(range(M), k))
                    random.shuffle(L)
                    while i < min(P, len(L)):
                        cw = torch.tensor([sum(weight[list(c)]) for c in L])
                        c_idx = torch.argmax(cw).item()
                        c = list(L[c_idx])
                        p = float(random.randint(0,1))
                        z_[i,:] = torch.tensor(p).repeat(M)
                        z_[i,c] =  torch.tensor(1-p).repeat(len(c))
                        weight[list(c)] = torch.tensor([1/(1+1/el.item()) for el in weight[list(c)]])
                        i += 1
                    k += 1
            
            # Random coal
            z_[i:, :] = torch.empty( max(0, num_samples-i), M).random_(2)
            return z_

       # Favour features - special coalitions don't study node's effect
        elif regu < 0.5:
            M = self.M - self.F
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples//2, M) # case k = 0 
            i = 2
            k = 1
            P = int(num_samples * 9/10)
            while i < P and k <= min(args_K, M-1): 

                # All coalitions of order k can be sampled 
                if i + 2 * scipy.special.comb(M,k) <= P:
                    L = list(combinations(range(M), k))
                    for c in L:
                        z_[i,c] = torch.zeros(k)
                        z_[i+1,c] = torch.ones(k)
                        i += 2
                    k += 1
            
                # All coalitions of order k cannot be sampled
                else: 
                    weight = torch.ones(M)
                    L = list(combinations(range(M), k))
                    random.shuffle(L)
                    while i < min(P, len(L)):
                        cw = torch.tensor([sum(weight[list(c)]) for c in L])
                        c_idx = torch.argmax(cw).item()
                        c = list(L[c_idx])
                        p = float(random.randint(0,1))
                        z_[i,:] = torch.tensor(p).repeat(M)
                        z_[i,c] =  torch.tensor(1-p).repeat(len(c))
                        weight[list(c)] = torch.tensor([1/(1+1/el.item()) for el in weight[list(c)]])
                        i += 1
                    k += 1
            
            # Random coal
            z_[i:, :] = torch.empty( max(0, num_samples-i), M).random_(2)
            return z_

  def Smarter(self, num_samples, args_K, *unused):
        """ Smart Mask generator
        Nodes and features are considered together but separately

        Args:
            num_samples ([int]): total number of coalitions z_
            args_K: max size of coalitions favoured in sampling 

        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        # Define empty and full coalitions
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples//2, self.M)
        i = 2
        k = 1
        # Loop until all samples are created
        while i < num_samples:
            # Look at each feat/nei individually if have enough sample
            # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
            if i + 2 * self.M < num_samples and k == 1:
                z_[i:i+self.M, :] = torch.ones(self.M, self.M)
                z_[i:i+self.M, :].fill_diagonal_(0)
                z_[i+self.M:i+2*self.M, :] = torch.zeros(self.M, self.M)
                z_[i+self.M:i+2*self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1

            else:
                # Split in two number of remaining samples
                # Half for specific coalitions with low k and rest random samples
                samp = i + 9*(num_samples - i)//10
                while i < samp and k <= args_K:
                    # Sample coalitions of k1 neighbours or k1 features without repet and order.
                    L = list(combinations(range(self.F), k)) + \
                        list(combinations(range(self.F, self.M), k))
                    random.shuffle(L)
                    L = L[:samp+1]

                    for j in range(len(L)):
                        # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                        z_[i, L[j]] = torch.zeros(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(
                                num_samples-i, self.M).random_(2)
                            return z_
                        # Coalitions (No nei, k feat) or (No feat, k nei)
                        z_[i, L[j]] = torch.ones(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(
                                num_samples-i, self.M).random_(2)
                            return z_
                    k += 1

                # Sample random coalitions
                z_[i:, :] = torch.empty(num_samples-i, self.M).random_(2)
                return z_
        return z_

  def Random(self, num_samples, *unused):
        """Sample masks randomly 

        """
        z_ = torch.empty(num_samples, self.M).random_(2)
        return z_

  def All(self, num_samples, *unsused):
        """Sample all possible 2^{F+N} coalitions (unordered, without replacement)

        Args:
            num_samples (int): min(all_coalition_cap, 2^{M+N}) or less

        [tensor]: dataset (2^{M+N} x self.M) where each row is in {0,1}^F x {0,1}^D
        """
        z_ = torch.zeros(num_samples, self.M)
        i = 0
        try:
            for k in range(0, self.M+1):
                L = list(combinations(range(0, self.M), k))
                for j in range(len(L)):
                    z_[i, L[j]] = torch.ones(k)
                    i += 1
        except IndexError:  # deal with boundary
            return z_
        return z_
  

  def shapley_kernel(self, s, M):
        """ Computes a weight for each newly created sample 

        Args:
            s (tensor): contains dimension of z for all instances
                (number of features + neighbours included)
            M (tensor): total number of features/nodes in dataset

        Returns:
                [tensor]: shapley kernel value for each sample
        """
        shapley_kernel = []

        for i in range(s.shape[0]):
            a = s[i].item()
            if a == 0 or a == M:
                # Enforce high weight on full/empty coalitions
                shapley_kernel.append(1000)
            elif scipy.special.binom(M, a) == float('+inf'):
                # Treat specific case - impossible computation
                shapley_kernel.append(1/ (M**2))
            else:
                shapley_kernel.append(
                    (M-1)/(scipy.special.binom(M, a)*a*(M-a)))

        shapley_kernel = np.array(shapley_kernel)
        shapley_kernel = np.where(shapley_kernel<1.0e-40, 1.0e-40,shapley_kernel)
        return torch.tensor(shapley_kernel)



  def compute_pred(self, graph_index, num_samples, D, z_, args_K, args_feat, true_pred):

        # Modify significanntly for our purpose 
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Graph Classification task

        Args:
            Variables are defined exactly as defined in explainer function
            Note that adjacency matrices are dense (square) matrices (unlike node classification)

        Returns:
            (tensor): f(z') - probability of belonging to each target classes, for all samples z'
            Dimension (N * C) where N is num_samples and C num_classses.
        """
        # Store discarded nodes (z_j=0) for each sample z 
        excluded_nei = {}
        for i in range(num_samples):
            # Excluded nodes' indexes 
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j])
            excluded_nei[i] = nodes_id
            # Dico with key = num_sample id, value = excluded neighbour index

        # Init
        # Note: legacy path that uses the older forward API; keep device handling explicit.
        fz = torch.zeros(num_samples)
        device = torch.device("cpu")
        data_x = self.data.get("x") if isinstance(self.data, dict) else getattr(self.data, "x", None)
        if torch.is_tensor(data_x):
            device = data_x.device
        adj = deepcopy(self.data.edge_index[graph_index])
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data.x[graph_index].shape[1])
        else: 
            num_atoms = int(self.data.get("num_atoms", self.data["x"].shape[0]))
            # Baseline mean over atom tokens only.
            av_feat_values = self.data.x[:num_atoms].mean(dim=0)
            #av_feat_values = np.mean(self.data.x[graph_index],axis=0)
        
        # Create new matrix A and X - for each sample ≈ reform z' from z
        num_atoms = int(self.data.get("num_atoms", self.data["x"].shape[0]))
        num_add_tokens = int(self.data.get("num_add_tokens", 0))
        num_players = num_atoms + num_add_tokens
        log_every_s = self.progress_log_every_s
        next_log = time.time() + log_every_s if log_every_s else None
        total = len(excluded_nei)
        for idx, (key, ex_nei) in enumerate(
            tqdm(excluded_nei.items(), file=sys.stdout), start=1
        ):
            if ex_nei and max(ex_nei) >= num_players:
                raise ValueError(
                    f"Excluded node index out of player range: max={max(ex_nei)}, "
                    f"num_players={num_players}"
                )

            # Change adj matrix
            A = deepcopy(adj)
            A[ex_nei, :] = 0
            A[:, ex_nei] = 0

            # Also change features of excluded nodes (optional)
            X = deepcopy(self.data.x[graph_index])
            for nei in ex_nei:
                X[nei] = av_feat_values

            # Apply model on (X,A) as input.
            with torch.no_grad():
                proba = self.model(
                    X.unsqueeze(0).to(device),
                    A.unsqueeze(0).to(device),
                ).exp()

            # Compute prediction
            fz[key] = proba[0][true_pred.item()].detach().cpu()
            if log_every_s and time.time() >= next_log:
                print(f"GraphSVX progress: {idx}/{total}", flush=True)
                next_log = time.time() + log_every_s

        return fz
  
  def graph_regression(self, graph_index, num_samples, D, z_, args_K, args_feat, true_pred, use_mask_only=False):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Graph Classification task

        Args:
            Variables are defined exactly as defined in explainer function
            Note that adjacency matrices are dense (square) matrices (unlike node classification)

        Returns:
            (tensor): f(z') - logit for target property, for all samples z'
            Dimension (N * C) where N is num_samples and C num_targets.
        """
        # Store discarded nodes (z_j=0) for each sample z 
        excluded_nei = {}
        for i in range(num_samples):
            # Excluded nodes' indexes 
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j])
            excluded_nei[i] = nodes_id
            # Dico with key = num_sample id, value = excluded neighbour index

        # Init 
        fz = torch.zeros(num_samples)
        device = self.data["x"].device if torch.is_tensor(self.data.get("x")) else torch.device("cpu")
        adj = deepcopy(self.data['edge_index'])  # removed graph_index capabilities here 
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data['x'].shape[1]) # removed graph_index capabilities here 
        else:
            num_atoms = int(self.data.get("num_atoms", self.data["x"].shape[0]))
            # Baseline mean should be computed over atom tokens only.
            # If additional tokens are excluded, attention masking handles their removal.
            av_feat_values = self.data['x'][:num_atoms].mean(dim=0)
            #av_feat_values = np.mean(self.data.x[graph_index],axis=0)
        
        # Create new matrix A and X - for each sample ≈ reform z' from z
        log_every_s = self.progress_log_every_s
        next_log = time.time() + log_every_s if log_every_s else None
        total = len(excluded_nei)
        for idx, (key, ex_nei) in enumerate(
            tqdm(excluded_nei.items(), file=sys.stdout), start=1
        ):
            device = self.data["x"].device if torch.is_tensor(self.data.get("x")) else next(self.model.parameters()).device

            # Change adj matrix
            A = deepcopy(adj)
            A[ex_nei, :] = 0
            A[:, ex_nei] = 0

            # Also change features of excluded nodes unless using mask-only ablation.
            X = self.data['x'].clone() # removed graph_index capabilities here 
            if not use_mask_only:
                for nei in ex_nei:
                    X[nei] = av_feat_values

            # Apply model on (X,A) as input.
            d = {'x': X, 'edge_index': A}
            # IMPORTANT: for this model family, adjacency is not used to build the transformer
            # attention mask. Only pass excluded nodes when using mask-only ablation.
            if use_mask_only:
                d["excluded_nodes"] = torch.as_tensor(ex_nei, dtype=torch.long)
                d["use_mask_only"] = True
            with torch.no_grad():
                d['x'] = d['x'].to(device)
                pred = self.model.infer_out(d, self.data['pred_index'], configs=self.data['config'])

            # Compute prediction
            pred_val = pred.detach().cpu()
            if pred_val.numel() == 1:
                pred_val = pred_val.squeeze(0)
            fz[key] = pred_val
            if log_every_s and time.time() >= next_log:
                print(f"GraphSVX progress: {idx}/{total}", flush=True)
                next_log = time.time() + log_every_s

        return fz
  
  def graph_classification(self, graph_index, num_samples, D, z_, args_K, args_feat, true_pred, use_mask_only=False):
        """ Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Graph Classification task

        Args:
            Variables are defined exactly as defined in explainer function
            Note that adjacency matrices are dense (square) matrices (unlike node classification)

        Returns:
            (tensor): f(z') - probability of belonging to each target classes, for all samples z'
            Dimension (N * C) where N is num_samples and C num_classses.
        """
        # Store discarded nodes (z_j=0) for each sample z 
        excluded_nei = {}
        for i in range(num_samples):
            # Excluded nodes' indexes 
            nodes_id = []
            for j in range(D):
                if z_[i, self.F+j] == 0:
                    nodes_id.append(self.neighbours[j])
            excluded_nei[i] = nodes_id
            # Dico with key = num_sample id, value = excluded neighbour index

        # Init 
        fz = torch.zeros(num_samples)
        adj = deepcopy(self.data['edge_index'])
        if args_feat == 'Null':
            av_feat_values = torch.zeros(self.data['x'].shape[1])
        else: 
            num_atoms = int(self.data.get("num_atoms", self.data["x"].shape[0]))
            # Baseline mean over atom tokens only.
            av_feat_values = self.data['x'][:num_atoms].mean(dim=0)
            #av_feat_values = np.mean(self.data.x[graph_index],axis=0)
        
        # Create new matrix A and X - for each sample ≈ reform z' from z
        log_every_s = self.progress_log_every_s
        next_log = time.time() + log_every_s if log_every_s else None
        total = len(excluded_nei)
        for idx, (key, ex_nei) in enumerate(
            tqdm(excluded_nei.items(), file=sys.stdout), start=1
        ):
            device = (
                self.data["x"].device
                if torch.is_tensor(self.data.get("x"))
                else next(self.model.parameters()).device
            )

            # Change adj matrix
            A = deepcopy(adj)
            A[ex_nei, :] = 0
            A[:, ex_nei] = 0

            # Also change features of excluded nodes unless using mask-only ablation.
            X = self.data['x'].clone()
            if not use_mask_only:
                for nei in ex_nei:
                    X[nei] = av_feat_values

            # Apply model on (X,A) as input.
            d = {'x': X, 'edge_index': A}
            if use_mask_only:
                d["excluded_nodes"] = torch.as_tensor(ex_nei, dtype=torch.long)
                d["use_mask_only"] = True
            with torch.no_grad():
                d['x'] = d['x'].to(device)
                _, feats = self.model.infer_out(
                    d,
                    self.data['pred_index'],
                    configs=self.data['config'],
                    return_features=True,
                )

            cls_feats = feats["cls_feats"]
            logits, is_binary = self.model.classification_head(cls_feats)
            if is_binary:
                logits = logits.squeeze(-1)
                probs_pos = torch.sigmoid(logits)
                pred_idx = int(self.data['pred_index'])
                if pred_idx == 0:
                    score = 1.0 - probs_pos
                elif pred_idx == 1:
                    score = probs_pos
                else:
                    raise ValueError(
                        f"Binary classification expects pred_index 0 or 1, got {pred_idx}."
                    )
                fz[key] = score.detach().cpu().squeeze(0)
            else:
                probs = torch.softmax(logits, dim=-1)
                pred_idx = int(self.data['pred_index'])
                if pred_idx >= probs.shape[-1]:
                    raise ValueError(
                        f"pred_index {pred_idx} out of range for {probs.shape[-1]} classes."
                    )
                fz[key] = probs[0, pred_idx].detach().cpu()
            if log_every_s and time.time() >= next_log:
                print(f"GraphSVX progress: {idx}/{total}", flush=True)
                next_log = time.time() + log_every_s

        return fz
  
  def WLS(self, z_, weights, fz, info):
        """ Weighted Least Squares Method
            Estimates shapley values via explanation model

        Args:
            z_ (tensor): binary vector representing the new instance
            weights ([type]): shapley kernel weights for z
            fz ([type]): prediction f(z') where z' is a new instance - formed from z and x

        Returns:
            [tensor]: estimated coefficients of our weighted linear regression - on (z, f(z'))
            Dimension (M * num_classes)
        """
        # Add constant term
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

        weights_np = weights.detach().cpu().numpy()
        z_np = z_.detach().cpu().numpy()
        fz_np = fz.detach().cpu().numpy()

        # WLS to estimate parameters
        try:
            tmp = np.linalg.inv(np.dot(np.dot(z_np.T, np.diag(weights_np)), z_np))
        except np.linalg.LinAlgError:  # matrix not invertible
            if info:
                print('WLS: Matrix not invertible')
            tmp = np.dot(np.dot(z_np.T, np.diag(weights_np)), z_np)
            tmp = np.linalg.inv(
                tmp + np.diag(10**(-5) * np.random.randn(tmp.shape[1])))

        phi = np.dot(tmp, np.dot(
            np.dot(z_np.T, np.diag(weights_np)), fz_np))

        # Test accuracy
        y_pred = z_np @ phi
        metrics = self._record_surrogate_metrics("WLS", fz_np, y_pred, weights_np)
        if info:
            print('r2: ', metrics["r2"])
            print('weighted r2: ', metrics["weighted_r2"])

        return phi[:-1], phi[-1]

  def WLS_Ridge(self, z_, weights, fz, info, alpha=1e-1):
        """ Weighted Least Squares (Ridge)
            Estimates shapley values via explanation model

        Args:
            z_ (tensor): binary vector representing the new instance
            weights ([type]): shapley kernel weights for z
            fz ([type]): prediction f(z') where z' is a new instance - formed from z and x
            alpha (float): ridge regularization strength

        Returns:
            [tensor]: estimated coefficients of ridge-regularized weighted regression
            Dimension (M * num_classes)
        """
        weights = weights.detach().cpu().numpy()
        z_ = z_.detach().cpu().numpy()
        fz = fz.detach().cpu().numpy()

        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(z_, fz, sample_weight=weights)
        y_pred = reg.predict(z_)

        metrics = self._record_surrogate_metrics("WLS_Ridge", fz, y_pred, weights)
        if info:
            print('weighted r2: ', metrics["weighted_r2"])
            print('r2: ', metrics["r2"])

        phi = reg.coef_
        base_value = reg.intercept_

        return phi, base_value
  
  def WLR_sklearn(self, z_, weights, fz, info):
        """ Weighted Linear Regression (sklearn)
            Estimates shapley values via explanation model

        Args:
            z_ (torch.tensor): dataset
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): predictions for z_ 

        Return:
            tensor: parameters of explanation model g
        """

        # Convert to numpy
        weights = weights.detach().numpy()
        z_ = z_.detach().numpy()
        fz = fz.detach().numpy()

        # Fit weighted linear regression
        reg = LinearRegression()
        reg.fit(z_, fz, weights)
        y_pred = reg.predict(z_)

        # Assess perf
        metrics = self._record_surrogate_metrics("WLR_sklearn", fz, y_pred, weights)
        if info:
            print('weighted r2: ', metrics["weighted_r2"])
            print('r2: ', metrics["r2"])

        # Coefficients
        phi = reg.coef_
        base_value = reg.intercept_

        return phi, base_value

  def WLR_Lasso(self, z_, weights, fz, info):
        """ Weighted Linear Regression (Lasso)
            Estimates shapley values via explanation model

        Args:
            z_ (torch.tensor): data
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): y data 
        
        Return:
            tensor: parameters of explanation model g
        
        """

        # Convert to numpy
        weights = weights.detach().numpy()
        z_ = z_.detach().numpy()
        fz = fz.detach().numpy()
        # Fit weighted linear regression
        reg = Lasso(alpha=0.01)
        # reg = Lasso()
        reg.fit(z_, fz, weights)
        y_pred = reg.predict(z_)
        # Assess perf
        metrics = self._record_surrogate_metrics("WLR_Lasso", fz, y_pred, weights)
        if info:
            print('weighted r2: ', metrics["weighted_r2"])
            print('r2: ', metrics["r2"])
        # Coefficients
        phi = reg.coef_
        base_value = reg.intercept_

        return phi, base_value

  def WTR_XGB(self, z_, weights, fz, info):
        """ Weighted Tree Regression (XGBoost)
            Estimates shapley values via explanation model

        Notes:
            - Uses a held-out split for fidelity reporting (train/test on z, not embeddings).
            - Requires optional dependency: xgboost.
            - TreeSHAP values are computed via XGBoost's pred_contribs for robustness.
        """
        try:
            import xgboost as xgb
        except Exception as exc:
            raise ImportError(
                "WTR_XGB requires the optional dependency `xgboost`."
            ) from exc

        Z = z_.detach().cpu().numpy()
        y = fz.detach().cpu().numpy()
        w = weights.detach().cpu().numpy()

        model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=0,
        )
        model.fit(Z, y, sample_weight=w)

        yhat = model.predict(Z)
        metrics = self._record_surrogate_metrics("WTR_XGB", y, yhat, w)
        if info:
            print("weighted r2: ", metrics["weighted_r2"])
            print("r2: ", metrics["r2"])

        # Explain the full coalition (all players present) using XGBoost's TreeSHAP.
        z0 = np.ones((1, Z.shape[1]), dtype=Z.dtype)
        dmat = xgb.DMatrix(z0)
        contribs = model.get_booster().predict(dmat, pred_contribs=True)
        phi0 = contribs[0, :-1].reshape(-1)
        base = float(contribs[0, -1])
        return phi0, base

  def compare_explanation_models(
        self,
        z_,
        weights,
        fz,
        models=None,
        test_frac=0.2,
        seed=0,
        info=True,
    ):
        """Compare surrogate models on in-sample and held-out fidelity.

        Args:
            z_ (torch.tensor): coalition masks.
            weights (torch.tensor): sample weights.
            fz (torch.tensor): target values.
            models (list[str]): model names to compare (e.g., ["WLS","WLS_Ridge","WLR_sklearn","WLR_Lasso","WTR_XGB"]).
            test_frac (float): fraction used for held-out evaluation.
            seed (int): RNG seed for the split.
            info (bool): print a compact report.

        Returns:
            dict: metrics keyed by model name.
        """
        if models is None:
            models = ["WLS", "WLS_Ridge", "WLR_sklearn", "WLR_Lasso"]

        Z = z_.detach().cpu().numpy()
        y = fz.detach().cpu().numpy()
        w = weights.detach().cpu().numpy()

        idx = np.arange(Z.shape[0])
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        split = int((1.0 - test_frac) * len(idx))
        tr, te = idx[:split], idx[split:]

        results = {}

        for name in models:
            if name == "WLS":
                Z_tr = np.concatenate([Z[tr], np.ones((len(tr), 1))], axis=1)
                Z_te = np.concatenate([Z[te], np.ones((len(te), 1))], axis=1)
                w_tr = np.sqrt(w[tr]).reshape(-1, 1)
                beta = np.linalg.pinv((Z_tr * w_tr).T @ (Z_tr * w_tr)) @ (
                    (Z_tr * w_tr).T @ (y[tr].reshape(-1, 1) * w_tr)
                )
                yhat_tr = (Z_tr @ beta).reshape(-1)
                yhat_te = (Z_te @ beta).reshape(-1)
            elif name == "WLS_Ridge":
                reg = Ridge(alpha=1e-3, fit_intercept=True)
                reg.fit(Z[tr], y[tr], sample_weight=w[tr])
                yhat_tr = reg.predict(Z[tr])
                yhat_te = reg.predict(Z[te])
            elif name == "WLR_sklearn":
                reg = LinearRegression()
                reg.fit(Z[tr], y[tr], sample_weight=w[tr])
                yhat_tr = reg.predict(Z[tr])
                yhat_te = reg.predict(Z[te])
            elif name == "WLR_Lasso":
                reg = Lasso(alpha=0.01)
                reg.fit(Z[tr], y[tr], sample_weight=w[tr])
                yhat_tr = reg.predict(Z[tr])
                yhat_te = reg.predict(Z[te])
            elif name == "WTR_XGB":
                try:
                    import xgboost as xgb
                except Exception as exc:
                    raise ImportError(
                        "WTR_XGB requires the optional dependency `xgboost`."
                    ) from exc
                reg = xgb.XGBRegressor(
                    n_estimators=800,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=0,
                )
                reg.fit(Z[tr], y[tr], sample_weight=w[tr])
                yhat_tr = reg.predict(Z[tr])
                yhat_te = reg.predict(Z[te])
            else:
                raise ValueError(f"Unknown surrogate model: {name}")

            results[name] = {
                "weighted_r2_train": r2_score(y[tr], yhat_tr, sample_weight=w[tr]),
                "r2_train": r2_score(y[tr], yhat_tr),
                "weighted_r2_heldout": r2_score(y[te], yhat_te, sample_weight=w[te]),
                "r2_heldout": r2_score(y[te], yhat_te),
            }

        if info:
            for name, metrics in results.items():
                print(
                    f"{name}: weighted r2 (train)={metrics['weighted_r2_train']:.4f}, "
                    f"r2 (train)={metrics['r2_train']:.4f}, "
                    f"weighted r2 (held-out)={metrics['weighted_r2_heldout']:.4f}, "
                    f"r2 (held-out)={metrics['r2_heldout']:.4f}"
                )

        return results

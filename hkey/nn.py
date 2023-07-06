import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.nn import init
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from hkey.utils import unpad_sequence
from fast_pytorch_kmeans import KMeans
import math
from typing import Tuple, List, Optional
from rpq.nn import RPQWeight
import numpy as np


class HKLinearDropIn(nn.Module):
    """Hierarchical key linear layer.

    Emulates the same functionality as nn.Linear, but with hierarchical clustering of weight columns
    into summary keys.  Then we can exploit sparsity in the query-key dot product to reduce the compute
    cost of the linear layer by only computing the projection for unique subset of query-keys that are above
    a certain threshold.
    
    """
    
    __constants__ = ['in_features', 'out_features', 'bias', 'method', 'threshold', 'sigma', 'n_clusters']
    in_features: int
    out_features: int
    n_clusters: int
    method: str
    threshold: float
    sigma: float
    bias: Optional[bool]
    weight: Tensor
    centroids: Tensor
    lengths: Tensor
    indices: Tensor

    def __init__(self, in_features: int, out_features: int,  n_clusters: int, 
                 method: str = "threshold", threshold: float = 0.5, sigma: float = 10.0,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_clusters = n_clusters
        self.method = method
        self.threshold = threshold
        self.sigma = sigma

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.register_buffer("ln_weight", torch.ones(self.n_clusters)*self.sigma)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.register_buffer("threshold_val", torch.tensor([self.threshold]))


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def init_hk(self, seed=0) -> None:
        # seed numpy and pytorch manually
        np.random.seed(seed)
        torch.manual_seed(seed)
        with torch.no_grad():
            km = KMeans(n_clusters=self.n_clusters)
            pred = km.fit_predict(self.weight)
        self.centroids = Parameter(km.centroids.T)
        indices = [torch.where(pred == i)[0].flatten() for i in range(km.n_clusters)]
        self.register_buffer("lengths", torch.tensor([len(i) for i in indices], dtype=torch.int64, device=self.weight.device))
        self.register_buffer("indices", pad_sequence(indices, batch_first=True, padding_value=-1))
        self.register_buffer("threshold_val", torch.tensor([self.threshold], device=self.weight.device))

    def scatter_tensor(self, input: Tensor, query_indices: Tensor, key_indices: Tensor, orig_batch_size: int) -> Tensor:
        key_ids = torch.empty(self.out_features, dtype=torch.long, device=input.device).fill_(input.shape[1])
        key_ids[key_indices.sort()[0]] = torch.arange(input.shape[1], device=input.device)

        query_ids = torch.empty(orig_batch_size, dtype=torch.long, device=input.device).fill_(input.shape[0])
        query_ids[query_indices.sort()[0]] = torch.arange(input.shape[0], device=input.device)

        input = F.embedding(key_ids, F.pad(input, (0, 1, 0, 0)).T).T
        input = F.embedding(query_ids, F.pad(input, (0, 0, 0, 1)))
        return input

    def key_search(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        dots = input@self.centroids
        dots = F.layer_norm(dots, (dots.shape[-1],), weight=self.ln_weight, bias=None, eps=1e-5).softmax(dim=-1)

        if self.method == "threshold":
            top_query_indices, top_cluster_indices = torch.where(dots > self.threshold_val)
            if top_query_indices.shape[0] != 0:
                top_query_indices = torch.unique(top_query_indices, dim=-1)
                top_cluster_indices = torch.unique(top_cluster_indices, dim=-1)
                top_key_indices = torch.concat(unpad_sequence(self.indices[top_cluster_indices], 
                                                        self.lengths[top_cluster_indices], 
                                                        batch_first=True))
            else:
                top_query_indices = torch.tensor([], dtype=torch.long, device=input.device)
                top_key_indices = torch.tensor([], dtype=torch.long, device=input.device)
        elif self.method == "max":
            top_query_indices = torch.tensor(torch.arange(input.shape[0]), dtype=torch.long, device=input.device)
            top_cluster_indices = torch.argmax(dots, dim=-1).unique()
            top_key_indices = torch.concat(unpad_sequence(self.indices[top_cluster_indices], 
                                                    self.lengths[top_cluster_indices], 
                                                    batch_first=True))
        return top_query_indices, top_key_indices

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        input = input.view(-1, input.shape[-1])
        n, _ = input.shape

        top_query_indices, key_indices = self.key_search(input)

        if top_query_indices.shape[0] != 0:
            # print(f"Surviving queries: {top_query_indices.shape[0]/input.shape[0]*100:.2f}%")
            # print(f"Surviving keys: {top_indices.shape[0]/self.out_features*100:.2f}%")
            print(f"Total non-zero elements: {((top_query_indices.shape[0]*key_indices.shape[0])/(n*self.out_features))*100:.2f}% \n")
            # Perform linear projection with the surviving queries and keys
            input = F.linear(input[top_query_indices], self.rpqweight(subset=key_indices), self.bias[key_indices] if self.bias is not None else None)
            input = self.scatter_tensor(input, top_query_indices, key_indices, n) # this is a hacky way to scatter the output using embeddings
            out = input
        else:
            print("No queries survived\n")
            # if no queries survive, then we just return a zero tensor
            input = torch.zeros((n, self.out_features), device=input.device, dtype=input.dtype, requires_grad=input.requires_grad)
            out = input
        return out.view(*shape[:-1], -1)
        
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, n_clusters={}, method={}, threshold={}, sigma={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.n_clusters, self.method, self.threshold, self.sigma)
    

class HKLinear(nn.Module):
    """Hierarchical key linear layer.

    Emulates the same functionality as nn.Linear, but with hierarchical clustering of weight columns
    into summary keys.  Then we can exploit sparsity in the query-key dot product to reduce the compute
    cost of the linear layer by only computing the projection for unique subset of query-keys that are above
    a certain threshold.
    
    """
    
    __constants__ = ['in_features', 'out_features', 'bias', 'method', 'threshold', 'sigma', 'n_clusters']
    in_features: int
    out_features: int
    n_clusters: int
    method: str
    threshold: float
    sigma: float
    bias: Optional[bool]
    weight: Tensor
    centroids: Tensor
    lengths: Tensor
    indices: Tensor

    def __init__(self, in_features: int, out_features: int,  n_clusters: int, 
                 method: str = "threshold", threshold: float = 0.5, sigma: float = 10.0,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_clusters = n_clusters
        self.method = method
        self.threshold = threshold
        self.sigma = sigma


        self.register_buffer("ln_weight", torch.ones(self.n_clusters)*self.sigma)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.register_buffer("threshold_val", torch.tensor([self.threshold]))
        self.init_weights()
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def init_weights(self) -> None:
        cluster_size = self.out_features // self.n_clusters
        self.centroids = torch.empty(self.n_clusters, self.in_features).normal_(0, 0.02)
        self.weights = self.centroids.repeat_interleave(cluster_size, dim=0) + torch.empty(self.out_features, self.in_features).normal_(0, 0.01)
        self.register_buffer("indices", torch.arange(self.out_features).reshape(self.n_clusters, cluster_size))


    def scatter_tensor(self, input: Tensor, query_indices: Tensor, key_indices: Tensor, orig_batch_size: int) -> Tensor:
        key_ids = torch.empty(self.out_features, dtype=torch.long, device=input.device).fill_(input.shape[1])
        key_ids[key_indices.sort()[0]] = torch.arange(input.shape[1], device=input.device)

        query_ids = torch.empty(orig_batch_size, dtype=torch.long, device=input.device).fill_(input.shape[0])
        query_ids[query_indices.sort()[0]] = torch.arange(input.shape[0], device=input.device)

        input = F.embedding(key_ids, F.pad(input, (0, 1, 0, 0)).T).T
        input = F.embedding(query_ids, F.pad(input, (0, 0, 0, 1)))
        return input

    def key_search(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        dots = input@self.centroids
        dots = F.layer_norm(dots, (dots.shape[-1],), weight=self.ln_weight, bias=None, eps=1e-5).softmax(dim=-1)

        if self.method == "threshold":
            top_query_indices, top_cluster_indices = torch.where(dots > self.threshold_val)
            if top_query_indices.shape[0] != 0:
                top_query_indices = torch.unique(top_query_indices, dim=-1)
                top_cluster_indices = torch.unique(top_cluster_indices, dim=-1)
                top_key_indices = self.indices
            else:
                top_query_indices = torch.tensor([], dtype=torch.long, device=input.device)
                top_key_indices = torch.tensor([], dtype=torch.long, device=input.device)
        elif self.method == "max":
            top_query_indices = torch.tensor(torch.arange(input.shape[0]), dtype=torch.long, device=input.device)
            top_cluster_indices = torch.argmax(dots, dim=-1).unique()
            top_key_indices = self.indices[top_cluster_indices]
        return top_query_indices, top_key_indices

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        input = input.view(-1, input.shape[-1])
        n, _ = input.shape

        top_query_indices, key_indices = self.key_search(input)

        if top_query_indices.shape[0] != 0:
            # print(f"Surviving queries: {top_query_indices.shape[0]/input.shape[0]*100:.2f}%")
            # print(f"Surviving keys: {top_indices.shape[0]/self.out_features*100:.2f}%")
            # print(f"Total non-zero elements: {((top_query_indices.shape[0]*key_indices.shape[0])/(n*self.out_features))*100:.2f}% \n")
            # Perform linear projection with the surviving queries and keys
            input = F.linear(input[top_query_indices], self.weight[key_indices], self.bias[key_indices] if self.bias is not None else None)
            input = self.scatter_tensor(input, top_query_indices, key_indices, n) # this is a hacky way to scatter the output using embeddings
            out = input
        else:
            # print("No queries survived\n")
            # if no queries survive, then we just return a zero tensor
            input = torch.zeros((n, self.out_features), device=input.device, dtype=input.dtype, requires_grad=input.requires_grad)
            out = input
        return out.view(*shape[:-1], -1)
        
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, n_clusters={}, method={}, threshold={}, sigma={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.n_clusters, self.method, self.threshold, self.sigma)


class HKRPQLinear(nn.Module):
    """Hierarchical key linear layer with RPQweights.

    Emulates the same functionality as nn.Linear, but with hierarchical clustering of weight columns
    into summary keys.  Then we can exploit sparsity in the input-summary dot product to reduce the compute
    cost of the linear layer by only computing the projection for unique subset of input-summary keys that 
    are above a certain threshold.
    
    """
    
    __constants__ = ['in_features', 'out_features', 'num_codebooks', 'method', 'n_clusters', 'threshold', 'sigma', 'shared_codebooks']
    in_features: int
    out_features: int
    num_codebooks: int
    method: str
    n_clusters: int
    threshold: float
    sigma: float
    shared_codebooks: bool
    bias: Optional[Tensor]
    codebooks: Tensor
    centroids: Tensor
    indices: Tensor
    lengths: Tensor
    threshold_val: Tensor
    ln_weight: Tensor

    def __init__(self, in_features: int, out_features: int, num_codebooks: int, method: str = "threshold",
                    n_clusters: Optional[int] = None, threshold: float = 0.98, sigma: float = 10.0,
                    shared_codebooks=False, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_codebooks = num_codebooks
        self.method = method
        self.n_clusters = num_codebooks if n_clusters is None else n_clusters
        self.threshold = threshold
        self.sigma = sigma
        self.shared_codebooks = shared_codebooks
        
        self.rpqweight = RPQWeight(self.num_codebooks, self.in_features//self.num_codebooks, self.out_features, 
                                   shared_codebooks=self.shared_codebooks, device=factory_kwargs['device'], 
                                   dtype=factory_kwargs['dtype'])
        self.register_buffer("ln_weight", torch.ones(self.n_clusters)*self.sigma)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.register_buffer("threshold_val", torch.tensor([self.threshold]))
        self.init_weights()
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if not self.shared_codebooks:
            init.normal_(self.rpqweight.codebooks, 0, 0.025)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.rpqweight())
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def init_weights(self) -> None:
        cluster_size = self.out_features // self.n_clusters
        centroids = torch.randint(0, 256, (self.n_clusters, self.num_codebooks), dtype=torch.uint8)
        codes = centroids.repeat_interleave(cluster_size, dim=0)
        perturb_select = torch.randint(0, self.num_codebooks, (self.out_features, max(1, round(0.25*self.num_codebooks))), device=codes.device)
        perturb_replace = torch.randint(0, 256, (self.out_features, max(1, round(0.25*self.num_codebooks))), device=codes.device, dtype=codes.dtype)
        codes.scatter_(1, perturb_select, perturb_replace)
        self.register_buffer("centroids", centroids.T) 
        self.register_buffer("indices", torch.arange(self.out_features).reshape(self.n_clusters, cluster_size))
        self.rpqweight.codes = codes.T

    def scatter_tensor(self, input: Tensor, query_indices: Tensor, key_indices: Tensor, orig_batch_size: int) -> Tensor:
        key_ids = torch.empty(self.out_features, dtype=torch.long, device=input.device).fill_(input.shape[1])
        key_ids[key_indices.sort()[0]] = torch.arange(input.shape[1], device=input.device)

        query_ids = torch.empty(orig_batch_size, dtype=torch.long, device=input.device).fill_(input.shape[0])
        query_ids[query_indices.sort()[0]] = torch.arange(input.shape[0], device=input.device)

        input = F.embedding(key_ids, F.pad(input, (0, 1, 0, 0)).T).T
        input = F.embedding(query_ids, F.pad(input, (0, 0, 0, 1)))
        return input

    def key_search(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        dots = input@self.rpqweight.expand(self.centroids, self.rpqweight.codebooks, subset=slice(None)).T
        dots = F.layer_norm(dots, (dots.shape[-1],), weight=self.ln_weight, bias=None, eps=1e-5).softmax(dim=-1)

        if self.method == "threshold":
            top_query_indices, top_cluster_indices = torch.where(dots > self.threshold_val)
            if top_query_indices.shape[0] != 0:
                top_query_indices = torch.unique(top_query_indices, dim=-1)
                top_cluster_indices = torch.unique(top_cluster_indices, dim=-1)
                top_key_indices = self.indices[top_cluster_indices].flatten()
            else:
                top_query_indices = torch.tensor([], dtype=torch.long, device=input.device)
                top_key_indices = torch.tensor([], dtype=torch.long, device=input.device)
        elif self.method == "max":
            top_query_indices = torch.arange(input.shape[0], dtype=torch.long, device=input.device)
            top_cluster_indices = torch.argmax(dots, dim=-1).unique()
            top_key_indices = self.indices[top_cluster_indices].flatten()
        return top_query_indices, top_key_indices

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        input = input.view(-1, input.shape[-1])
        n, _ = input.shape

        top_query_indices, key_indices = self.key_search(input)

        if top_query_indices.shape[0] != 0:
            # Perform linear projection with the surviving queries and keys
            input = F.linear(input[top_query_indices], self.rpqweight(subset=key_indices), self.bias[key_indices] if self.bias is not None else None)
            input = self.scatter_tensor(input, top_query_indices, key_indices, n) # this is a hacky way to scatter the output using embeddings
            out = input
        else:
            # if no queries survive, then we just return a zero tensor
            input = torch.zeros((n, self.out_features), device=input.device, dtype=input.dtype, requires_grad=input.requires_grad)
            out = input
        return out.view(*shape[:-1], -1)
        
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, num_codebooks={}, method={}, \
            n_clusters={}, threshold={}, sigma={} shared_codebooks={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.num_codebooks, self.method, 
            self.n_clusters, self.threshold, self.sigma, self.shared_codebooks
        )

    

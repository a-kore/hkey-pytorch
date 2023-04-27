import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.nn import init
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from fast_pytorch_kmeans import KMeans
import math
from typing import Tuple


class HKLinear(nn.Module):
    """Hierarchical Key Linear Layer.

    Emulates the same functionality as nn.Linear, but with a hierarchical key associative memory.
    
    """
        
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    n_clusters: int
    threshold: float
    weight: Tensor
    centroids: Tensor

    def __init__(self, in_features: int, out_features: int, n_clusters: int, 
                 threshold: float = 0.01, temperature: float = 0.1, bias: bool = True, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HKLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.temperature = temperature
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def init_clusters(self) -> None:
        with torch.no_grad():
            km = KMeans(n_clusters=self.n_clusters)
            pred = km.fit_predict(self.weight)
        self.centroids = Parameter(km.centroids)
        indices = [torch.where(pred == i)[0].flatten() for i in range(km.n_clusters)]
        self.register_buffer("lengths", torch.tensor([len(i) for i in indices], device=self.weight.device))
        self.register_buffer("indices", pad_sequence(indices, batch_first=True, padding_value=-1))

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        input = input.view(-1, input.shape[-1])
        n, _ = input.shape

        dots = (input@self.centroids.T).divide_(self.temperature).softmax(dim=-1)
        query_indices, cluster_indices = torch.where(dots > self.threshold)
        
        if query_indices.shape[0] != 0:
            top_query_indices = torch.unique(query_indices, dim=-1)
            top_cluster_indices = torch.unique(cluster_indices, dim=-1)
            top_indices = torch.concat(unpad_sequence(self.indices[top_cluster_indices], 
                                                    self.lengths[top_cluster_indices], 
                                                    batch_first=True))
            out = torch.zeros((n, self.out_features), device=input.device)
            out[top_query_indices] = out[top_query_indices].index_copy(1, top_indices, 
                                                                    F.linear(input[top_query_indices],
                                                                                self.weight[top_indices],
                                                                                self.bias[top_indices]))
        else:
            out = torch.zeros((n, self.out_features), device=input.device)
        return out.view(*shape[:-1], -1)
        

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_clusters={}, threshold={}'.format(
            self.in_features, self.out_features, self.n_clusters, self.threshold
        )
    
class HKLinear1D(nn.Module):
    """Hierarchical Key Linear Layer.

    Emulates the same functionality as nn.Linear, but with a hierarchical key associative memory.
    
    """
        
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    n_clusters: int
    threshold: float
    temperature: float
    weight: Tensor
    centroids: Tensor

    def __init__(self, in_features: int, out_features: int, n_clusters: int, 
                 threshold: float = 0.01, temperature: float = 0.1, bias: bool = True, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HKLinear1D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.temperature = temperature
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def init_clusters(self) -> None:
        with torch.no_grad():
            km = KMeans(n_clusters=self.n_clusters)
            pred = km.fit_predict(self.weight)
        self.centroids = Parameter(km.centroids)
        indices = [torch.where(pred == i)[0].flatten() for i in range(km.n_clusters)]
        self.register_buffer("lengths", torch.tensor([len(i) for i in indices], device=self.weight.device))
        self.register_buffer("indices", pad_sequence(indices, batch_first=True, padding_value=-1))

    def forward(self, input: Tensor) -> Tensor:

        dots = (input@self.centroids.T).divide_(self.temperature).softmax(dim=-1)
        query_indices, cluster_indices = torch.where(dots > self.threshold)
        
        if query_indices.shape[0] != 0:
            top_query_indices = torch.unique(query_indices, dim=-1)
            top_cluster_indices = torch.unique(cluster_indices, dim=-1)
            top_indices = torch.concat(unpad_sequence(self.indices[top_cluster_indices], 
                                                    self.lengths[top_cluster_indices], 
                                                    batch_first=True))
            out = torch.zeros((input.shape[0], self.out_features), device=input.device)
            out[top_query_indices] = out[top_query_indices].index_copy(1, top_indices, 
                                                                    F.linear(input[top_query_indices],
                                                                                self.weight[top_indices],
                                                                                self.bias[top_indices]))
        else:
            out = torch.zeros((input.shape[0], self.out_features), device=input.device)
        return out
        

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_clusters={}, threshold={}'.format(
            self.in_features, self.out_features, self.n_clusters, self.threshold
        )
    

class HKLinear2D(nn.Module):
    """Hierarchical Key Linear Layer.

    Emulates the same functionality as nn.Linear, but with a hierarchical key associative memory.
    
    """
        
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    n_clusters: int
    threshold: float
    weight: Tensor
    centroids: Tensor

    def __init__(self, in_features: int, out_features: int, n_clusters: int, 
                 threshold: float = 0.01, temperature: float = 0.1, bias: bool = True, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HKLinear2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.temperature = temperature
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def init_clusters(self) -> None:
        with torch.no_grad():
            km = KMeans(n_clusters=self.n_clusters)
            pred = km.fit_predict(self.weight)
        self.centroids = Parameter(km.centroids)
        indices = [torch.where(pred == i)[0].flatten() for i in range(km.n_clusters)]
        self.register_buffer("lengths", torch.tensor([len(i) for i in indices], device=self.weight.device))
        self.register_buffer("indices", pad_sequence(indices, batch_first=True, padding_value=-1))

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        dim0, dim1 = input.shape[:-1]
        input = input.view(-1, input.shape[-1])
        n, _ = input.shape

        dots = (input@self.centroids.T).divide_(self.temperature).softmax(dim=-1)
        query_indices, cluster_indices = torch.where(dots > self.threshold)
        
        if query_indices.shape[0] != 0:
            top_query_indices = torch.unique(query_indices, dim=-1)
            top_cluster_indices = torch.unique(cluster_indices, dim=-1)
            top_indices = torch.concat(unpad_sequence(self.indices[top_cluster_indices], 
                                                    self.lengths[top_cluster_indices], 
                                                    batch_first=True))
            out = torch.zeros((n, self.out_features), device=input.device)
            out[top_query_indices] = out[top_query_indices].index_copy(1, top_indices, 
                                                                    F.linear(input[top_query_indices],
                                                                                self.weight[top_indices],
                                                                                self.bias[top_indices]))
        else:
            out = torch.zeros((n, self.out_features), device=input.device)
        return out.view(dim0, dim1, -1)
        

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, n_clusters={}, threshold={}'.format(
            self.in_features, self.out_features, self.n_clusters, self.threshold
        )
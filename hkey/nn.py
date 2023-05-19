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
from typing import Tuple, List
from routing_transformer.routing_transformer import Kmeans


class HKLinear(nn.Module):
    """Hierarchical key linear layer.

    Emulates the same functionality as nn.Linear, but with hierarchical clustering of weight columns
    into summary keys.  Then we can exploit sparsity in the query-key dot product to reduce the compute
    cost of the linear layer by only computing the projection for unique subset of query-keys that are above
    a certain threshold.
    
    """
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    num_codebooks: int
    n_clusters: int
    threshold: float
    temperature: float
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, n_clusters: int, 
                 threshold: float = 1.0e-4, temperature: float = 1.0,
                 bias: bool = True, device=None, dtype=None) -> None:
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

        self.centroids = Parameter(torch.empty((n_clusters, in_features), **factory_kwargs))
        self.register_buffer("lengths", torch.empty(n_clusters, **factory_kwargs))
        self.register_buffer("indices", torch.empty(n_clusters, out_features//n_clusters **factory_kwargs))
        self.threshold = Parameter(torch.tensor(self.threshold, **factory_kwargs))


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    
    def init_hk(self) -> None:
        with torch.no_grad():
            km = KMeans(n_clusters=self.n_clusters)
            pred = km.fit_predict(self.rpqweight())
        self.centroids = Parameter(km.centroids.T)
        indices = [torch.where(pred == i)[0].flatten() for i in range(km.n_clusters)]
        self.lengths = torch.tensor([len(i) for i in indices], device=self.weight.device)
        self.indices = pad_sequence(indices, batch_first=True, padding_value=-1)
        self.threshold = Parameter(torch.tensor(self.threshold, device=self.weight.device))

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        input = input.view(-1, input.shape[-1])

        dots = (input@self.centroids.T).divide_(self.temperature).softmax(dim=-1)
        query_indices, cluster_indices = torch.where(dots > self.threshold)
        
        if query_indices.shape[0] != 0:
            top_query_indices = torch.unique(query_indices, dim=-1)
            top_cluster_indices = torch.unique(cluster_indices, dim=-1)
            top_indices = torch.concat(unpad_sequence(self.indices[top_cluster_indices], 
                                                    self.lengths[top_cluster_indices], 
                                                    batch_first=True))
            input = F.linear(input[top_query_indices], self.weight[top_indices], self.bias[top_indices] if self.bias is not None else None)
            num_features = input.shape[-1]
            input = F.pad(input, (0, self.out_features - num_features))
            indices = torch.arange(self.out_features, device=top_indices.device)
            zero_indices = indices[torch.isin(indices, top_indices, invert=True)]
            input[:, top_indices] = input[:, 0:num_features].clone()
            input[:, zero_indices] = 0
            out = input
        else:
            num_features = input.shape[-1]
            input = F.pad(input, (0, self.out_features - num_features))
            input.zero_()
            out = input
        return out.view(*shape[:-1], -1)
        

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, n_clusters={}, threshold={}, temperature={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.n_clusters, self.threshold, self.temperature
        )


class HKLinear1D(nn.Module):
    """Hierarchical Key Linear Layer 1D (batch, dim)
    
    Torch.jit.script doesn't like a variable number of arguments, so we need to make separate classes for
    ND inputs. (https://github.com/pytorch/pytorch/issues/29637)

    Emulates the same functionality as nn.Linear, but with hierarchical clustering of weight columns
    into summary keys.  Then we can exploit sparsity in the query-key dot product to reduce the compute
    cost of the linear layer by only computing the projection for unique subset of query-keys that are above
    a certain threshold.
    
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
                 threshold: float = 1.0e-4, temperature: float = 1.0, bias: bool = True, 
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
        shape = input.shape
        input = input.view(-1, input.shape[-1])

        dots = (input@self.centroids.T).divide_(self.temperature).softmax(dim=-1)
        query_indices, cluster_indices = torch.where(dots > self.threshold)
        
        if query_indices.shape[0] != 0:
            top_query_indices = torch.unique(query_indices, dim=-1)
            top_cluster_indices = torch.unique(cluster_indices, dim=-1)
            top_indices = torch.concat(unpad_sequence(self.indices[top_cluster_indices], 
                                                    self.lengths[top_cluster_indices], 
                                                    batch_first=True))
            input = F.linear(input[top_query_indices], self.weight[top_indices], self.bias[top_indices] if self.bias is not None else None)
            num_features = input.shape[-1]
            input = F.pad(input, (0, self.out_features - num_features))
            indices = torch.arange(self.out_features, device=top_indices.device)
            zero_indices = indices[torch.isin(indices, top_indices, invert=True)]
            input[:, top_indices] = input[:, 0:num_features].clone()
            input[:, zero_indices] = 0
            out = input
        else:
            num_features = input.shape[-1]
            input = F.pad(input, (0, self.out_features - num_features))
            input.zero_()
            out = input
        return out.view(*shape[:-1], -1)
    

class HKLinear2D(nn.Module):
    """Hierarchical Key Linear Layer 2D (batch, seq, dim)  
    
    Torch.jit.script doesn't like a variable number of arguments, so we need to make separate classes for
    ND inputs. (https://github.com/pytorch/pytorch/issues/29637)

    Emulates the same functionality as nn.Linear, but with hierarchical clustering of weight columns
    into summary keys.  Then we can exploit sparsity in the query-key dot product to reduce the compute
    cost of the linear layer by only computing the projection for unique subset of query-keys that are above
    a certain threshold.
    
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    n_clusters: int
    threshold: float
    weight: Tensor
    centroids: Tensor

    def __init__(self, in_features: int, out_features: int, n_clusters: int, 
                 threshold: float = 1.0e-4, temperature: float = 1.0, bias: bool = True, 
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

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        input = input.view(-1, input.shape[-1])

        dots = (input@self.centroids.T).divide_(self.temperature).softmax(dim=-1)
        query_indices, cluster_indices = torch.where(dots > self.threshold)
        
        if query_indices.shape[0] != 0:
            top_query_indices = torch.unique(query_indices, dim=-1)
            top_cluster_indices = torch.unique(cluster_indices, dim=-1)
            top_indices = torch.concat(unpad_sequence(self.indices[top_cluster_indices], 
                                                    self.lengths[top_cluster_indices], 
                                                    batch_first=True))
            input = F.linear(input[top_query_indices], self.weight[top_indices], self.bias[top_indices] if self.bias is not None else None)
            num_features = input.shape[-1]
            input = F.pad(input, (0, self.out_features - num_features))
            indices = torch.arange(self.out_features, device=top_indices.device)
            zero_indices = indices[torch.isin(indices, top_indices, invert=True)]
            input[:, top_indices] = input[:, 0:num_features].clone()
            input[:, zero_indices] = 0
            out = input
        else:
            num_features = input.shape[-1]
            input = F.pad(input, (0, self.out_features - num_features))
            input.zero_()
            out = input
        return out.view(*shape[:-1], -1)


class HKRPQLinearDropIn(nn.Module):
    """Applies linear transformation to the incoming data.
    
       This module supports the drop-in replacement of nn.Linear.
    
    """
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    num_codebooks: int
    n_clusters: int
    threshold: float
    temperature: float
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, num_codebooks: int,
                  n_clusters: int = None, threshold: float = 1.0e-4, temperature: float = 1.0,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HKRPQLinearDropIn, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_codebooks = num_codebooks
        self.n_clusters = n_clusters if n_clusters is not None else num_codebooks
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

    
    def init_hk(self) -> None:
        with torch.no_grad():
            km = KMeans(n_clusters=self.n_clusters)
            pred = km.fit_predict(self.rpqweight())
        self.centroids = Parameter(km.centroids)
        indices = [torch.where(pred == i)[0].flatten() for i in range(km.n_clusters)]
        self.register_buffer("lengths", torch.tensor([len(i) for i in indices], device=self.rpqweight.codebooks.device))
        self.register_buffer("indices", pad_sequence(indices, batch_first=True, padding_value=-1))

    def init_rpq(self):
        self.rpqweight = RPQWeight(self.num_codebooks, 
                                self.in_features//self.num_codebooks, 
                                self.out_features)
        self.rpqweight.init_rpq(self.weight)
        del self.weight

    def init_hkrpq(self) -> None:
        self.init_rpq()
        self.init_hk()
        self.threshold = Parameter(torch.tensor(self.threshold, device=self.rpqweight.codebooks.device))

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        input = input.view(-1, input.shape[-1])

        dots = (input@self.centroids.T).divide_(self.temperature).softmax(dim=-1)
        query_indices, cluster_indices = torch.where(dots > self.threshold)
        
        if query_indices.shape[0] != 0:
            top_query_indices = torch.unique(query_indices, dim=-1)
            top_cluster_indices = torch.unique(cluster_indices, dim=-1)
            top_indices = torch.concat(unpad_sequence(self.indices[top_cluster_indices], 
                                                    self.lengths[top_cluster_indices], 
                                                    batch_first=True))
            input = F.linear(input[top_query_indices], self.rpqweight(subset=top_indices), self.bias[top_indices] if self.bias is not None else None)
            num_features = input.shape[-1]
            input = F.pad(input, (0, self.out_features - num_features))
            indices = torch.arange(self.out_features, device=top_indices.device)
            zero_indices = indices[torch.isin(indices, top_indices, invert=True)]
            input[:, top_indices] = input[:, 0:num_features].clone()
            input[:, zero_indices] = 0
            out = input
        else:
            num_features = input.shape[-1]
            input = F.pad(input, (0, self.out_features - num_features))
            input.zero_()
            out = input
        return out.view(*shape[:-1], -1)
        

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, num_codebooks={}, n_clusters={}, threshold={}, temperature={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.num_codebooks, self.n_clusters, self.threshold, self.temperature
        )
    

class HKRPQLinear(nn.Module):
    """Applies linear transformation to the incoming data.
    
       This module supports the drop-in replacement of nn.Linear.
    
    """
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    num_codebooks: int
    n_clusters: int
    threshold: float
    temperature: float

    def __init__(self, in_features: int, out_features: int, num_codebooks: int,
                  n_clusters: int = None, threshold: float = 1.0e-4, temperature: float = 1.0,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HKRPQLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_codebooks = num_codebooks
        self.n_clusters = n_clusters if n_clusters is not None else num_codebooks
        self.threshold = threshold
        self.temperature = temperature

        self.init_hkrpq()

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.rpqweight.codebooks, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.rpqweight())
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    
    def init_hk(self) -> None:
        with torch.no_grad():
            km = KMeans(n_clusters=self.n_clusters)
            pred = km.fit_predict(self.rpqweight())
        self.centroids = Parameter(km.centroids)
        indices = [torch.where(pred == i)[0].flatten() for i in range(km.n_clusters)]
        self.register_buffer("lengths", torch.tensor([len(i) for i in indices], device=self.rpqweight.codebooks.device))
        self.register_buffer("indices", pad_sequence(indices, batch_first=True, padding_value=-1))

    def init_rpq(self):
            self.rpqweight = RPQWeight(self.num_codebooks, 
                                    self.in_features//self.num_codebooks, 
                                    self.out_features)

    def init_hkrpq(self) -> None:
        self.init_rpq()
        self.init_hk()
        self.threshold = Parameter(torch.tensor(self.threshold, device=self.rpqweight.codebooks.device))

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        input = input.view(-1, input.shape[-1])

        dots = (input@self.centroids.T).divide_(self.temperature).softmax(dim=-1)
        query_indices, cluster_indices = torch.where(dots > self.threshold)
        
        if query_indices.shape[0] != 0:
            top_query_indices = torch.unique(query_indices, dim=-1)
            top_cluster_indices = torch.unique(cluster_indices, dim=-1)
            top_indices = torch.concat(unpad_sequence(self.indices[top_cluster_indices], 
                                                    self.lengths[top_cluster_indices], 
                                                    batch_first=True))
            input = F.linear(input[top_query_indices], self.rpqweight(subset=top_indices), self.bias[top_indices] if self.bias is not None else None)
            num_features = input.shape[-1]
            input = F.pad(input, (0, self.out_features - num_features))
            indices = torch.arange(self.out_features, device=top_indices.device)
            zero_indices = indices[torch.isin(indices, top_indices, invert=True)]
            input[:, top_indices] = input[:, 0:num_features].clone()
            input[:, zero_indices] = 0
            out = input
        else:
            num_features = input.shape[-1]
            input = F.pad(input, (0, self.out_features - num_features))
            input.zero_()
            out = input
        return out.view(*shape[:-1], -1)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, num_codebooks={}, n_clusters={}, split= {}, threshold={}, temperature={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.num_codebooks, self.n_clusters, self.split, self.threshold, self.temperature
        )


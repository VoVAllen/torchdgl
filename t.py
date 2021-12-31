import torch
from torch import ops
import dgl
from dgl.data import CoraGraphDataset
torch.set_num_threads(1)
torch.classes.load_library("libTorchDGLGraph.so")
torch.classes.load_library("libCustomOp.so")
# print(torch.classes.loaded_libraries)

g = CoraGraphDataset()[0]
src, dst = g.edges()

s = torch.classes.my_classes.TorchDGLGraph(src, dst)
TorchDGLGraph = torch.classes.my_classes.TorchDGLGraph


def do_in_degrees(s: TorchDGLGraph, vids: torch.Tensor, u: torch.Tensor):
    return ops.custom.indegree_op_with_autograd(s, vids, u)


scripted_func = torch.jit.script(do_in_degrees)
print(scripted_func.graph)
print("##########DGL#########")
u = torch.tensor([1,3,5,7], dtype=torch.float, requires_grad=True)
output = do_in_degrees(s, torch.tensor([1,3,5,7]), u)
print(output, u)
output.sum().backward()
print(u.grad)

print("##########TorchScript DGL#########")
u = torch.tensor([1,3,5,7], dtype=torch.float, requires_grad=True)
output = do_in_degrees(s, torch.tensor([1,3,5,7]), u)
print(output, u)
output.sum().backward()
print(u.grad)

scripted_func.save("in_degree.pt")
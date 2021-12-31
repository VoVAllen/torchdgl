#include <torchdglgraph/custom_op.h>
#include <torchdglgraph/torch_dgl_graph.h>

#include <c10/util/intrusive_ptr.h>


TORCH_LIBRARY(custom, m) {
  m.def("indegree_op_with_autograd(__torch__.torch.classes.my_classes.TorchDGLGraph g, Tensor vids, Tensor u) -> Tensor", [](
    const c10::intrusive_ptr<TorchDGLGraph> &tg,
    torch::Tensor vids,
    torch::Tensor u) {
      return CustomOpAutogradFunction::apply(*tg, vids, u);
    });
}

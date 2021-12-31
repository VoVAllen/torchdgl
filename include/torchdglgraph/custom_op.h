#include <torchdglgraph/torch_dgl_graph.h>


struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      TorchDGLGraph tg,
      torch::Tensor vids,
      torch::Tensor u) {
    ctx->saved_data["vids"] = vids;
    torch::Tensor ret = tg.InDegrees(vids);
    // the indegree output is a int tensor, which do not contains grad and can't support backend.
    torch::Tensor tt = u + ret;
    return tt;
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_output) {
    torch::Tensor vids = ctx->saved_data["vids"].toTensor();
    torch::Tensor indegree = grad_output[0];
    // here I just try to return a value for grad. If no grad required, can just return torch::Tensor().
    torch::autograd::variable_list output = {torch::Tensor(), torch::Tensor(), vids};
    return output;
  }
};

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main()
{
        namespace tf = tensorflow;
        tf::Scope root = tf::Scope::NewRootScope();

        auto A = tf::ops::Const(root, {{3.f, 3.f}, {-1.f, 0.f}});

        tf::ClientSession session(root);
        std::vector<tf::Tensor> outputs;
        TF_CHECK_OK(session.Run({A}, &outputs));

        LOG(INFO) << outputs[0].matrix<float>();
}

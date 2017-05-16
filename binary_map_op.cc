#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tf = tensorflow;

tf::REGISTER_OP("ZeroOut")
        .Input("to_zero: int32")
        .Output("zeroed: int32")
        .SetShapeFn([](tf::tensorflow::shape_inference::InferenceContext *c) {
                    c->set_output(0, c->input(0));
                    return Status::OK();
        });

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("BinaryMap")
        .Input("radius: int32")
        .Output("binary_map: int32")
        .SetShapeFn([](shape_inference::InferenceContext *c) {
                c->set_output(0, c->input(0));
                return Status::OK();
        });

struct BinaryMapOp : OpKernel {
        explicit BinaryMapOp(OpKernelConstruction *context) : OpKernel(context) {}

        void Compute(OpKernelContext *context) override {
                const Tensor& input_tensor = context->input(0);
                auto input = input_tensor.flat<int32>();

                Tensor *output_tensor = nullptr;
                OP_REQUIRES_OK(context,
                               context->allocate_output(0,
                                                        input_tensor.shape(),
                                                        &output_tensor));

                auto output = output_tensor->flat<int32>();
                const int32_t N = input.size();
                for (int32_t i = 1;
                     i < N;
                     ++i) {
                        output(i) = 0;
                }

                if (N > 0)
                        output(0) = input(0);
        }
};

REGISTER_KERNEL_BUILDER(Name("BinaryMap").Device(DEVICE_CPU), BinaryMapOp);

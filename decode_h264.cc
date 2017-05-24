#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

namespace tf = tensorflow;

static tf::Status
DecodeH264ShapeFn(tf::shape_inference::InferenceContext *c)
{
        tf::shape_inference::ShapeHandle unused;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

        c->set_output(0,
                      c->MakeShape({tf::shape_inference::InferenceContext::kUnknownDim,
                                    tf::shape_inference::InferenceContext::kUnknownDim,
                                    tf::shape_inference::InferenceContext::kUnknownDim,
                                    3}));

        return tf::Status::OK();
}

REGISTER_OP("DecodeH264")
        .Input("video: string")
        .Attr("width: int = 256")
        .Attr("height: int = 256")
        .Attr("num_frames: int = 16")
        .Attr("seek_distance: float = 0.0")
        .Output("frames: uint8")
        .SetShapeFn(DecodeH264ShapeFn)
        .Doc(R"doc(
Decode an h264 video to a sequence of frames, stored as one uint8 tensor.
)doc");

struct DecodeH264Op : tf::OpKernel {
        explicit DecodeH264Op(tf::OpKernelConstruction *context) : tf::OpKernel(context) {
                OP_REQUIRES_OK(context, context->GetAttr("width", &width_));
                OP_REQUIRES_OK(context, context->GetAttr("height", &height_));
                OP_REQUIRES_OK(context, context->GetAttr("num_frames",
                                                         &num_frames_));
                OP_REQUIRES_OK(context, context->GetAttr("seek_distance",
                                                         &seek_distance_));
        }

        void Compute(tf::OpKernelContext *context) override {
                int32_t status;

                const tf::Tensor& contents = context->input(0);
                OP_REQUIRES(context,
                            tf::TensorShapeUtils::IsScalar(contents.shape()),
                            tf::errors::InvalidArgument("contents must be scalar, got shape ",
                                                        contents.shape().DebugString()));

                const char *cmd_fmt = "ffmpeg -i temp.mp4 -pix_fmt rgb24 -s %dx%d "
                                      "-ss %.2f -c:v rawvideo -map 0:v -f rawvideo "
                                      "-vframes %d "
                                      "pipe:1 "
                                      "-loglevel warning";
                char cmd_str[128];
                int32_t num_written = snprintf(cmd_str,
                                               sizeof(cmd_str),
                                               cmd_fmt,
                                               width_,
                                               height_,
                                               seek_distance_,
                                               num_frames_);
                assert((num_written != EOF) &&
                       (num_written < sizeof(cmd_str)));

                FILE *stream = popen(cmd_str, "r");
                OP_REQUIRES(context,
                            stream != NULL,
                            tf::errors::Internal("popen() failed."));

                tf::Tensor *output = nullptr;
                context->allocate_output(0,
                                         tf::TensorShape({num_frames_, height_, width_, 3}),
                                         &output);

		uint32_t total_count = 0;
		uint32_t count = 0;
                do {
                        count = fread(output->flat<uint8_t>().data() + total_count,
                                      1,
                                      width_,
                                      stream);
                        total_count += count;
                } while ((count > 0) && (total_count < output->flat<uint8_t>().size()));

                status = pclose(stream);
                OP_REQUIRES(context,
                            status != -1,
                            tf::errors::Internal("pclose() failed."));
        }

private:
        int32_t width_;
        int32_t height_;
        int32_t num_frames_;
        float seek_distance_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeH264").Device(tf::DEVICE_CPU), DecodeH264Op);

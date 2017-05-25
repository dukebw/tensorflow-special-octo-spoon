#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <sys/wait.h>
#include <stdio.h>
#include <stdarg.h>
#include <unistd.h>
#include <assert.h>

namespace tf = tensorflow;

static void
close_pipe(int32_t pipe_fd)
{
        int32_t status = close(pipe_fd);
        assert(status != -1);
}

static void
create_pipe(int32_t pipe_fd[2])
{
        int32_t status = pipe(pipe_fd);
        assert(status != -1);
}

static void
dup_pipe(int32_t pipe_fd, int32_t fd_to_dup)
{
        if (pipe_fd != fd_to_dup) {
                int32_t status = dup2(pipe_fd, fd_to_dup);
                assert(status != -1);

                status = close(pipe_fd);
                assert(status != -1);
        }
}

static void
format_string(char *str, uint32_t max_size, const char *fmt, ...)
{
        va_list va_args;
        int32_t num_written;

        va_start(va_args, fmt);
        num_written = vsnprintf(str, max_size, fmt, va_args);
        va_end(va_args);

        assert((num_written != EOF) && (num_written < max_size));
}

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
        explicit DecodeH264Op(tf::OpKernelConstruction *context) : tf::OpKernel(context)
        {
                OP_REQUIRES_OK(context, context->GetAttr("width", &width_));
                OP_REQUIRES_OK(context, context->GetAttr("height", &height_));
                OP_REQUIRES_OK(context, context->GetAttr("num_frames",
                                                         &num_frames_));
                OP_REQUIRES_OK(context, context->GetAttr("seek_distance",
                                                         &seek_distance_));
        }

        void Compute(tf::OpKernelContext *context) override
        {
                int32_t status;

                const tf::Tensor& contents = context->input(0);
                OP_REQUIRES(context,
                            tf::TensorShapeUtils::IsScalar(contents.shape()),
                            tf::errors::InvalidArgument("contents must be scalar, got shape ",
                                                        contents.shape().DebugString()));

                tf::Tensor *output = nullptr;
                context->allocate_output(0,
                                         tf::TensorShape({num_frames_, height_, width_, 3}),
                                         &output);

                const int32_t NUM_PIPES = 2;
                const int32_t PARENT_WRITE_PIPE = 0;
                const int32_t PARENT_READ_PIPE = 1;
                const int32_t READ_FD = 0;
                const int32_t WRITE_FD = 1;
                int32_t pipes[NUM_PIPES][2];
                create_pipe(pipes[PARENT_READ_PIPE]);
                create_pipe(pipes[PARENT_WRITE_PIPE]);

                pid_t ffmpeg_pid = fork();
                assert(ffmpeg_pid >= 0);

                switch (ffmpeg_pid) {
                case 0:
                        {
                                close_pipe(pipes[PARENT_READ_PIPE][READ_FD]);
                                close_pipe(pipes[PARENT_WRITE_PIPE][WRITE_FD]);

                                dup_pipe(pipes[PARENT_READ_PIPE][WRITE_FD],
                                         STDOUT_FILENO);
                                dup_pipe(pipes[PARENT_WRITE_PIPE][READ_FD],
                                         STDIN_FILENO);

                                char size_str[16];
                                format_string(size_str,
                                              sizeof(size_str),
                                              "%dx%d",
                                              width_,
                                              height_);

                                char seek_str[8];
                                format_string(seek_str,
                                              sizeof(seek_str),
                                              "%.2f",
                                              seek_distance_);

                                char num_frames_str[8];
                                format_string(num_frames_str,
                                              sizeof(num_frames_str),
                                              "%d",
                                              num_frames_);

                                const char *argv[] = {
                                        "ffmpeg",
                                        "-i", "pipe:0",
                                        "-pix_fmt", "rgb24",
                                        "-s", size_str,
                                        "-ss", seek_str,
                                        "-c:v", "rawvideo",
                                        "-map", "0:v",
                                        "-f", "rawvideo",
                                        "-vframes", num_frames_str,
                                        "pipe:1",
                                        "-loglevel", "warning",
                                        NULL
                                };

                                execvp("ffmpeg", (char * const *)argv);
                                assert(false);
                        }
                        break;
                default:
                        {
                                close_pipe(pipes[PARENT_READ_PIPE][WRITE_FD]);
                                close_pipe(pipes[PARENT_WRITE_PIPE][READ_FD]);
                        }
                        break;
                }

                uint32_t count;
                pid_t write_child_pid = fork();
                assert(write_child_pid >= 0);

                switch (write_child_pid) {
                case 0:
                        {
                                close_pipe(pipes[PARENT_READ_PIPE][READ_FD]);

                                const tf::StringPiece input = contents.scalar<tf::string>()();

                                count = write(pipes[PARENT_WRITE_PIPE][WRITE_FD],
                                              input.data(),
                                              input.size());
                                fprintf(stderr, "written count: %u\n", count);

                                _exit(EXIT_SUCCESS);
                        }
                        break;
                default:
                        {
                                close_pipe(pipes[PARENT_WRITE_PIPE][WRITE_FD]);

                                uint32_t total_count = 0;
                                uint32_t count;
                                do {
                                        count = read(pipes[PARENT_READ_PIPE][READ_FD],
                                                     output->flat<uint8_t>().data() + total_count,
                                                     output->flat<uint8_t>().size() - total_count);
                                        total_count += count;
                                } while ((count > 0) && (total_count < output->flat<uint8_t>().size()));

                                if (total_count != output->flat<uint8_t>().size()) {
                                        fprintf(stderr, "read count: %u requested count: %lu\n",
                                                total_count,
                                                output->flat<uint8_t>().size());
                                }
                        }
                        break;
                }

                pid_t child_pid = waitpid(write_child_pid, NULL, 0);
                assert(child_pid != -1);
        }

private:
        int32_t width_;
        int32_t height_;
        int32_t num_frames_;
        float seek_distance_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeH264").Device(tf::DEVICE_CPU), DecodeH264Op);

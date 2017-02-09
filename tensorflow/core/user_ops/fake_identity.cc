/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example Op.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

using namespace tensorflow;

REGISTER_OP("FakeIdentity")
    .Input("x_in: T")
    .Output("x_out: T")
    .Output("sf_proxy: T")
    .Output("bits_proxy: T")
    .Attr("T: type")
    .Doc(R"doc(
Faking an identity operation for convenient gradient quantization.
)doc");

class FakeIdentityOp : public OpKernel {
 public:
  explicit FakeIdentityOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    

    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
    //printf("1\n");
    TensorShape sf_proxy_shape({});
    //sf_proxy_shape.AddDim(1);
    //printf("2\n");

    TensorShape bits_proxy_shape({});
    //bits_proxy_shape.AddDim(1);
    //printf("3\n");

    /*const Tensor sf_proxy(DT_FLOAT, TensorShape({}));
    const Tensor bits_proxy(DT_FLOAT, TensorShape({}));
    std::cout<<sf_proxy.scalar<float>()()<<std::endl;
    sf_proxy.scalar<float>()() = 0.0f;
    bits_proxy.scalar<float>()() = 0.0f;

    const Tensor* sf_proxy_ptr = &sf_proxy;
    const Tensor* bits_proxy_ptr = &bits_proxy;
    std::cout<<"!"<<std::endl;
    context->set_output(1, *sf_proxy_ptr);
    context->set_output(2, *bits_proxy_ptr);
    */
    Tensor* sf_proxy = nullptr;
    Tensor* bits_proxy = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, sf_proxy_shape, &sf_proxy));
    OP_REQUIRES_OK(context, context->allocate_output(2, bits_proxy_shape, &bits_proxy));
    //printf("4\n");
    //auto d_sf_proxy = sf_proxy->tensor<float, 1>();
    //auto d_bits_proxy = bits_proxy->tensor<float, 1>();
    auto d_sf_proxy = sf_proxy->scalar<int>();
    auto d_bits_proxy = bits_proxy->scalar<int>();

    

    //std::cout<<d_sf_proxy.size()<<" "<<d_sf_proxy.rank()<<std::endl;
    //std::cout<<d_sf_proxy.size()<<std::endl;
    //std::cout<<d_sf_proxy()<<" " << d_bits_proxy()<<std::endl;
    d_sf_proxy() = 0;
    d_bits_proxy() = 0;
    //printf("!!!????\n");
  }

  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("FakeIdentity").Device(DEVICE_CPU), FakeIdentityOp);

/**
#define REGISTER_GPU_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("FakeIdentity").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      FakeIdentityOp);                                                       \

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
//REGISTER_GPU_KERNEL(bfloat16);
REGISTER_GPU_KERNEL(float);

#undef REGISTER_GPU_KERNEL

**/
#if GOOGLE_CUDA
// A special GPU kernel for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("FakeIdentity")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("x_in")        \
                              .HostMemory("x_out")	\
			      .HostMemory("sf_proxy")	\
			      .HostMemory("bits_proxy")       \
                              .TypeConstraint<type>("T"), \
                          FakeIdentityOp);                    \

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(bool);
REGISTER_GPU_HOST_KERNEL(float);

#undef REGISTER_GPU_HOST_KERNEL

#endif

/*
* Copyright 2015-2017 Philippe Tillet
* Copyright (c) 2017, Intel Corporation
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifdef HAVE_OPENCL

#include <sstream>
#include "opencl_kernels_core.hpp"
#include "opencv2/core/opencl/runtime/opencl_clblas.hpp"
#include "opencv2/core/opencl/runtime/opencl_core.hpp"

namespace cv
{

static bool intel_gpu_gemm(
    UMat A, Size sizeA,
    UMat B, Size sizeB,
    UMat D, Size sizeD,
    double alpha, double beta,
    bool atrans, bool btrans)
{
    CV_UNUSED(sizeB);

    int M = sizeD.height, N = sizeD.width, K = ((atrans)? sizeA.height : sizeA.width);

    std::string kernelName;
    bool ret = true;

    size_t lx = 8, ly = 4;
    size_t dx = 4, dy = 8;

    if(!atrans && !btrans)
    {

        if (M % 32 == 0 && N % 32 == 0 && K % 16 == 0)
        {
            kernelName = "intelblas_gemm_buffer_NN_sp";
        }
        else
        {
            kernelName = "intelblas_gemm_buffer_NN";
        }
    }
    else if(atrans && !btrans)
    {
        kernelName = "intelblas_gemm_buffer_TN";
    }
    else if(!atrans && btrans)
    {
        kernelName = "intelblas_gemm_buffer_NT";
        ly = 16;
        dx = 1;
    }
    else
    {
        kernelName = "intelblas_gemm_buffer_TT";
    }

    const size_t gx = (size_t)(N + dx - 1) / dx;
    const size_t gy = (size_t)(M + dy - 1) / dy;

    size_t local[] = {lx, ly, 1};
    size_t global[] = {(gx + lx - 1) / lx * lx, (gy + ly - 1) / ly * ly, 1};

    int stride = (M * N < 1024 * 1024) ? 10000000 : 256;

    ocl::Queue q;
    String errmsg;
    const ocl::Program program = ocl::Context::getDefault().getProg(ocl::core::intel_gemm_oclsrc, "", errmsg);

    if(!atrans && btrans)
    {
        ocl::Kernel k(kernelName.c_str(), program);
        if (k.empty())
        {
            return false;
        }

        k.args(ocl::KernelArg::PtrReadOnly(A),
               (int) (A.offset / sizeof(float)),
               ocl::KernelArg::PtrReadOnly(B),
               (int) (B.offset / sizeof(float)),
               ocl::KernelArg::PtrWriteOnly(D),
               (int) (D.offset / sizeof(float)),
               M, N, K,
               (float)alpha,
               (float)beta,
               (int)(A.step / sizeof(float)),
               (int)(B.step / sizeof(float)),
               (int)(D.step / sizeof(float))
        );

        ret = k.run(2, global, local, false, q);
    }
    else
    {
        for(int start_index = 0; start_index < K; start_index += stride)
        {
             ocl::Kernel k(kernelName.c_str(), program);
             k.args(ocl::KernelArg::PtrReadOnly(A),
                    (int) (A.offset / sizeof(float)),
                    ocl::KernelArg::PtrReadOnly(B),
                    (int) (B.offset / sizeof(float)),
                    ocl::KernelArg::PtrWriteOnly(D),
                    (int) (D.offset / sizeof(float)),
                    M, N, K,
                    (float)alpha,
                    (float)beta,
                    (int)(A.step / sizeof(float)),
                    (int)(B.step / sizeof(float)),
                    (int)(D.step / sizeof(float)),
                    (int) start_index,                          // 14 start_index
                    stride);

            ret = k.run(2, global, local, false, q);
            if (!ret) return ret;
        }
    }

    return ret;
}

} // namespace cv

#endif

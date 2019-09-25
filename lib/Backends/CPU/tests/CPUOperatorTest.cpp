/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {
    "less_float16Cases/0",
    "less_int64Cases/0",
    "ResizeNearest_Float/0",
    "ResizeNearest_Float16/0",
    "ResizeNearest_Int8/0",
    "ResizeNearest_Int16/0",
    "ResizeNearest_Int32/0",
    "replaceNaN_Float16/0",
    "Logit_Float16/0",
    "FP16Add/0",
    "FP16Matmul/0",
    "batchedReduceAdd_Float16/0",
    "batchedReduceZeroDimResult_Float16/0",
    "batchedReduceAddWithAxis_Float16/0",
    "ReluSimple_Float16/0",
    "PReluSimple_Float16/0",
    "GatherDataFloat16IdxInt32/0",
    "GatherDataFloat16IdxInt64/0",
    "GatherRangesDataFloat16IdxInt32/0",
    "GatherRangesDataFloat16IdxInt64/0",
    "FP16Transpose2Dims/0",
    "Transpose3Dims_Float16/0",
    "ArithAdd_int32_t/0",
    "ArithAdd_int64_t/0",
    "ArithAdd_float16_t/0",
    "ArithSub_int32_t/0",
    "ArithSub_int64_t/0",
    "ArithSub_float16_t/0",
    "ArithMul_int32_t/0",
    "ArithMul_int64_t/0",
    "ArithMul_float16_t/0",
    "ArithMax_int32_t/0",
    "ArithMax_int64_t/0",
    "ArithMax_float16_t/0",
    "ArithMin_int32_t/0",
    "ArithMin_int64_t/0",
    "ArithMin_float16_t/0",
    "convTest_Float16/0",
    "FP16Max/0",
    "concatVectors_Int32/0",
    "concatVectors_Float16/0",
    "concatVectorsRepeated_Int32/0",
    "concatVectorsRepeated_Float16/0",
    "sliceVectors_Float16/0",
    "sliceConcatVectors_Float16/0",
    "ExpandDims_Float16/0",
    "Split_Float16/0",
    "Fp16Splat/0",
    "GroupConv3D/0",
    "NonCubicPaddingConv3D/0",
    "FP16AvgPool/0",
    "AdaptiveAvgPool/0",
    "FP16AdaptiveAvgPool/0",
    "Int8AdaptiveAvgPool/0",
    "AdaptiveAvgPoolNonSquare/0",
    "FP16MaxPool/0",
    "NonCubicKernelConv3D/0",
    "NonCubicKernelConv3DQuantized/0",
    "NonCubicStrideConv3D/0",
    "FP16BatchAdd/0",
    "Sigmoid_Float16/0",
    "testBatchAdd_Float16/0",
    "SparseLengthsSum_Float16/0",
    "SparseLengthsSumI8/0",
    "SparseLengthsWeightedSum_1D_Float16/0",
    "SparseLengthsWeightedSum_2D_Float16/0",
    "SparseLengthsWeightedSumI8/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16/0",
    "RowwiseQuantizedSparseLengthsSum_Float16_AccumFloat/0",
    "RowwiseQuantizedSparseLengthsSum_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsSum_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSparseLengthsSum_Float16_AccumFloat16/0",
    "SparseToDenseMask1/0",
    "SparseToDenseMask2/0",
    "FP16Reshape/0",
    "sliceReshape_Float16/0",
    "Flatten_Float16Ty/0",
    "Bucketize/0",
    "FP16SoftMax/0",
    "BatchOneHotDataFloat/0",
    "BatchOneHotDataFloat16/0",
    "BatchOneHotDataInt64/0",
    "BatchOneHotDataInt32/0",
    "BatchOneHotDataInt8/0",
    "dotProduct1D_Float16/0",
    "dotProduct2D_Float16/0",
    "BatchBoxCox_Float/0",
    "BatchBoxCox_Float16/0",
    "ConvertFrom_FloatTy_To_Float16Ty/0",
    "ConvertFrom_FloatTy_To_Int32ITy/0",
    "ConvertFrom_FloatTy_To_Int64ITy/0",
    "ConvertFrom_Float16Ty_To_FloatTy/0",
    "ConvertFrom_Float16Ty_To_Float16Ty/0",
    "ConvertFrom_Float16Ty_To_Int32ITy/0",
    "ConvertFrom_Float16Ty_To_Int64ITy/0",
    "ConvertFrom_Int32ITy_To_FloatTy/0",
    "ConvertFrom_Int32ITy_To_Float16Ty/0",
    "ConvertFrom_Int32ITy_To_Int64ITy/0",
    "ConvertFrom_Int64ITy_To_FloatTy/0",
    "ConvertFrom_Int64ITy_To_Float16Ty/0",
    "ConvertFrom_Int64ITy_To_Int32ITy/0",
    "ConvertFrom_FloatTy_To_Float16Ty_AndBack/0",
    "ConvertFrom_FloatTy_To_Int32ITy_AndBack/0",
    "ConvertFrom_FloatTy_To_Int64ITy_AndBack/0",
    "ConvertFrom_Float16Ty_To_FloatTy_AndBack/0",
    "ConvertFrom_Float16Ty_To_Float16Ty_AndBack/0",
    "ConvertFrom_Float16Ty_To_Int32ITy_AndBack/0",
    "ConvertFrom_Float16Ty_To_Int64ITy_AndBack/0",
    "ConvertFrom_Int32ITy_To_FloatTy_AndBack/0",
    "ConvertFrom_Int32ITy_To_Float16Ty_AndBack/0",
    "ConvertFrom_Int64ITy_To_FloatTy_AndBack/0",
    "ConvertFrom_Int64ITy_To_Float16Ty_AndBack/0",
    "ConvertFrom_Int64ITy_To_Int32ITy_AndBack/0",
    "BasicDivNetFloatVsInt8/0",
    "BasicAddNetFloatVsFloat16/0",
    "BasicSubNetFloatVsFloat16/0",
    "BasicMulNetFloatVsFloat16/0",
    "BasicDivNetFloatVsFloat16/0",
    "BasicMaxNetFloatVsFloat16/0",
    "BasicMinNetFloatVsFloat16/0",
    "Int16ConvolutionDepth10/0",
    "Int16ConvolutionDepth8/0",
    "FP16ConvolutionDepth10/0",
    "FP16ConvolutionDepth8/0",
    "FC_Float16/0",
    "Tanh_Float16/0",
    "Exp_Float16/0",
    "rowwiseQuantizedSLWSTest/0",
};
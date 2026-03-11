/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <optional>

#include "ATen/core/TensorBody.h"
#include "extensions.h"
#include "pybind.h"
#include "util.h"
#include "common.h"

std::vector<py::object> fused_multi_quantize(std::vector<py::handle> input_list,
                                             std::optional<std::vector<py::handle>> output_list,
                                             std::vector<py::handle> quantizer_list,
                                             transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;
  std::vector<transformer_engine::TensorWrapper> tensor_input_list;
  std::vector<transformer_engine::TensorWrapper> tensor_output_list;
  std::vector<py::object> py_output_objects_list;
  auto none = py::none();

  if(output_list == std::nullopt && transformer_engine::pytorch::detail::IsMTFP8QParams(quantizer_list[0].ptr())) {
    const size_t N = input_list.size();
    
    std::vector<std::unique_ptr<Quantizer>> quantizers;
    quantizers.reserve(N);

    bool can_not_share = false;
    for (int i = 0; i < N; ++i) {
        quantizers.emplace_back(convert_quantizer(quantizer_list[i]));        
        auto quantizer = static_cast<MTFP8Quantizer*>(quantizers[i].get());
        can_not_share = can_not_share || quantizer->block_m != quantizer->block_n;        
    }
    
    bool is_same_dim_n = true;
    std::vector<std::vector<int64_t>> data_shapes, row_scale_shapes, col_scale_shapes;
    for (int i = 0; i < N; ++i) {
        tensor_input_list.emplace_back(makeTransformerEngineTensor(input_list[i], none));
        
        const NVTEShape shape = tensor_input_list[i].shape();
        int64_t dim_m = 1, dim_n = shape.data[shape.ndim - 1];
        for (int j = 0; j < shape.ndim - 1; ++j) dim_m *= shape.data[j];
        data_shapes.emplace_back(std::vector<int64_t>{dim_m, dim_n});
        
        auto quantizer = static_cast<MTFP8Quantizer*>(quantizers[i].get());
        if(quantizer->rowwise_usage) {
            int64_t sinv0 = (dim_m + quantizer->block_m - 1) / quantizer->block_m;
            int64_t sinv1 = (dim_n + quantizer->block_n - 1) / quantizer->block_n;
            row_scale_shapes.emplace_back(std::vector<int64_t>{sinv0, sinv1});
        }
        if(quantizer->columnwise_usage && can_not_share) {
            int64_t sinv0 = (dim_m + quantizer->block_n - 1) / quantizer->block_n;
            int64_t sinv1 = (dim_n + quantizer->block_m - 1) / quantizer->block_m;
            col_scale_shapes.emplace_back(std::vector<int64_t>{sinv0, sinv1});
        }

        if(i > 0 && is_same_dim_n && dim_n != data_shapes[0][1]) is_same_dim_n = false;
    }
    
    std::vector<at::Tensor> row_data, row_scale, col_data, col_scale;
    if (is_same_dim_n) {
        if(row_scale_shapes.size() > 0) {
            int64_t data_total = 0, scale_total = 0;
            std::vector<int64_t> data_slices(N), scale_slices(N);
            for (int i = 0; i < N; ++i) {
                data_slices[i] = data_shapes[i][0];
                data_total += data_slices[i];
                scale_slices[i] = row_scale_shapes[i][0];
                scale_total += scale_slices[i];
            }
            row_data = at::empty({data_total, data_shapes[0][1]}, at::TensorOptions().device(torch::kPrivateUse1).dtype(torch::kUInt8)).split(data_slices, 0);
            row_scale = at::empty({scale_total, row_scale_shapes[0][1]}, at::TensorOptions().device(torch::kPrivateUse1).dtype(torch::kFloat)).split(scale_slices, 0);
        }

        if(col_scale_shapes.size() > 0) {
            int64_t data_total = 0, scale_total = 0;
            std::vector<int64_t> data_slices(N), scale_slices(N);
            for (int i = 0; i < N; ++i) {
                data_slices[i] = data_shapes[i][0];
                data_total += data_slices[i];
                scale_slices[i] = col_scale_shapes[i][0];
                scale_total += scale_slices[i];
            }
            col_data = at::empty({data_total, data_shapes[0][1]}, at::TensorOptions().device(torch::kPrivateUse1).dtype(torch::kUInt8)).split(data_slices, 0);
            col_scale = at::empty({scale_total, col_scale_shapes[0][1]}, at::TensorOptions().device(torch::kPrivateUse1).dtype(torch::kFloat)).split(scale_slices, 0);
        }
    } else {
        // TODO: Support varying dimension sizes - currently unimplemented as no use case exists.
        // Currently, operations require all tensors to have the same size in dim_n.
        NVTE_CHECK(is_same_dim_n);
    }

    using namespace pybind11::literals;
    for (int i = 0; i < N; ++i) {
        auto quantizer = static_cast<MTFP8Quantizer*>(quantizers[i].get());
        transformer_engine::TensorWrapper tensor(NVTE_MTFP8_BLOCK_SCALING);
        std::vector<size_t> data_shape(data_shapes[i].begin(), data_shapes[i].end());
        
        at::Tensor _row_data, _row_scale;
        if(quantizer->rowwise_usage) {
            _row_data = row_data[i];
            _row_scale = row_scale[i];
            std::vector<size_t> scale_shape(row_scale_shapes[i].begin(), row_scale_shapes[i].end());
            tensor.set_rowwise_data(_row_data.data_ptr(), quantizer->dtype, data_shape);
            tensor.set_rowwise_scale_inv(_row_scale.data_ptr(), DType::kFloat32, scale_shape);
        }
        
        at::Tensor _col_data, _col_scale;
        if(quantizer->columnwise_usage && can_not_share) {
            _col_data = col_data[i];
            _col_scale = col_scale[i];
            std::vector<size_t> scale_shape(col_scale_shapes[i].begin(), col_scale_shapes[i].end());
            tensor.set_columnwise_data(_col_data.data_ptr(), quantizer->dtype, data_shape);
            tensor.set_columnwise_scale_inv(_col_scale.data_ptr(), DType::kFloat32, scale_shape);
        }
        quantizer->set_quantization_params(&tensor);
        tensor_output_list.emplace_back(std::move(tensor));
        
        py::object ret;
        if (quantizer->internal) {
            py::handle MTFP8TensorClass(reinterpret_cast<PyObject*>(MTFP8TensorBasePythonClass));
            ret = MTFP8TensorClass("rowwise_data"_a = _row_data,
                                "columnwise_data"_a = _col_data,
                                "rowwise_scale_inv"_a = _row_scale,
                                "columnwise_scale_inv"_a = _col_scale,
                                "fp8_dtype"_a = quantizer->dtype,
                                "quantizer"_a = quantizer->quantizer);
        } else {
            py::handle MTFP8TensorClass(reinterpret_cast<PyObject*>(MTFP8TensorPythonClass));
            ret = MTFP8TensorClass("shape"_a = data_shapes[i],
                                "dtype"_a = GetATenDType(otype),
                                "rowwise_data"_a = _row_data,
                                "columnwise_data"_a = _col_data,
                                "rowwise_scale_inv"_a = _row_scale,
                                "columnwise_scale_inv"_a = _col_scale,
                                "fp8_dtype"_a = quantizer->dtype,
                                "quantizer"_a = quantizer->quantizer);
        }
        py_output_objects_list.emplace_back(std::move(ret));
    }

  } else {
    // create TE tensors from input
    for (int i = 0; i < input_list.size(); i++) {
        auto input_tensor = makeTransformerEngineTensor(input_list[i], none);
        const NVTEShape input_shape = input_tensor.shape();

        transformer_engine::TensorWrapper output_tensor;

        if (output_list == std::nullopt) {
            std::unique_ptr<Quantizer> quantizer = convert_quantizer(quantizer_list[i]);
            std::vector<size_t> output_shape(input_shape.data, input_shape.data + input_shape.ndim);
            py::object o;
            std::tie(output_tensor, o) = quantizer->create_tensor(output_shape, otype);
            py_output_objects_list.push_back(o);
        } else {
            output_tensor = makeTransformerEngineTensor((*output_list)[i], quantizer_list[i]);
        }
        // if (input_tensor.numel() == 0) continue;

        tensor_output_list.emplace_back(std::move(output_tensor));
        tensor_input_list.emplace_back(std::move(input_tensor));
    }
  }


  // Check tensor lists
  NVTE_CHECK(tensor_output_list.size() == tensor_input_list.size(),
             "Number of input and output tensors must match");

  // Choose implementation
  // Note: Currently only have fused kernel for block wise FP8 cast-transpose
  bool with_fused_kernel = true;
  for (int i = 0; i < tensor_output_list.size(); i++) {
    const auto tensor = tensor_output_list[i].data();
    if (nvte_tensor_scaling_mode(tensor) != NVTE_MTFP8_BLOCK_SCALING) {
      with_fused_kernel = false;
      break;
    }
    // if (nvte_tensor_columnwise_data(tensor) == nullptr) {
    //   with_fused_kernel = false;
    //   break;
    // }
  }

  // Launch TE kernel
  if (with_fused_kernel) {
    std::vector<NVTETensor> nvte_tensor_input_list;
    std::vector<NVTETensor> nvte_tensor_output_list;
    for (int i = 0; i < tensor_input_list.size(); i++) {
        if(tensor_input_list[i].numel() == 0) continue;
        nvte_tensor_input_list.emplace_back(tensor_input_list[i].data());
        nvte_tensor_output_list.emplace_back(tensor_output_list[i].data());
    }

    nvte_multi_cast_transpose(nvte_tensor_input_list.size(), nvte_tensor_input_list.data(),
                              nvte_tensor_output_list.data(), at::musa::getCurrentMUSAStream());
  } else {
    for (int i = 0; i < tensor_input_list.size(); i++) {
      if(tensor_input_list[i].numel() == 0) continue;
      nvte_quantize(tensor_input_list[i].data(), tensor_output_list[i].data(),
                    at::musa::getCurrentMUSAStream());
    }
  }
  return py_output_objects_list;
}

at::Tensor fp8_transpose(at::Tensor input, transformer_engine::DType otype,
                         std::optional<at::Tensor> output) {
  using namespace transformer_engine::pytorch;

  const auto dim = input.dim();
  NVTE_CHECK(dim >= 2, "Need at least 2D tensor to transpose.");

  if (input.dim() > 2) {
    input = input.view({-1, input.size(dim - 1)});
  }

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  at::Tensor out;
  if (output.has_value()) {
    out = *output;
  } else {
    out = allocateTorchTensor(input.size(1), input.size(0), DType::kByte);
  }
  if (M == 0 || N == 0) return out;

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(out.data_ptr(), {N, M}, otype);

  nvte_transpose(input_cu.data(), output_cu.data(), at::musa::getCurrentMUSAStream());

  return out;
}

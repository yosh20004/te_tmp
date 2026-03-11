/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_
#define TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_

#include "transformer_engine/transformer_engine.h"
#include "../common.h"
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>


namespace transformer_engine {

typedef void *NVTEGroupedTensor;

struct GroupedTensor {
 public:
  /* EXPERIMENTAL FEATURE AND SUBJECT TO CHANGE. */
  /*
  Grouped tensor is a collection of tensors with different shapes but the same dtype and scaling mode

  Shape Representation:
  - logical_shape: 2D shape representing the conceptual layouy, i.e. the shape when member tensors are flattened to 2D and stacked together (REQUIRED)
    + When all_same_shape(): [num_tensors * M, N] where each tensor is (M, N)
    + When varying_first_dim(): [~sum_of_first_dims, N] where N is common
    + When varying_last_dim(): [M, ~sum_of_last_dims] where M is common
    + When varying_both_dims(): [1, total_elements] (fully flattened)

  - first_dims and last_dims are OPTIONAL (empty if dimension is uniform)
    + Empty first_dims: all tensors have the same first dimension
    + Empty last_dims: all tensors have the same last dimension
    + Both empty: all tensors have identical shapes
    + Both set: each tensor has unique shape (first_dims[i], last_dims[i])

  Data Layout:
  - ALL data fields are stored as 1D flattened arrays (data, columnwise_data, scale_inv, etc.)
  - logical_shape provides the conceptual 2D interpretation
  - All data is stored on device in contiguous layout
  */

  SimpleTensor data;
  SimpleTensor columnwise_data;
  SimpleTensor scale_inv;
  SimpleTensor columnwise_scale_inv;
  SimpleTensor amax;
  SimpleTensor columnwise_amax;
  SimpleTensor scale;  // for FP8-DS only

  NVTEScalingMode scaling_mode;
  size_t num_tensors;

  // Shape information (OPTIONAL - empty if dimension is uniform across all tensors)
  // first_dims[i] = first dimension of tensor i (empty if all tensors have same first dim)
  // last_dims[i] = last dimension of tensor i (empty if all tensors have same last dim)
  SimpleTensor first_dims;  // Device pointer to int64_t array of length num_tensors (or empty)
  SimpleTensor last_dims;   // Device pointer to int64_t array of length num_tensors (or empty)

  // Offsets for indexing into contiguous 1D layout (OPTIONAL - not needed if all_same_shape())
  // tensor_offsets[i] = element offset to start of tensor i (cumulative sum of numel for tensors 0..i-1)
  // Usage: tensor_i_ptr = (char*)data.dptr + tensor_offsets[i] * element_size
  // If empty and all_same_shape(): offset[i] = i * M * N (where M, N are common dimensions)
  SimpleTensor tensor_offsets;  // Device pointer to int64_t array of length num_tensors (or empty)

  // Logical shape: conceptual 2D shape of the grouped data (REQUIRED)
  // Represents how the 1D flattened data should be interpreted as 2D
  // Always 2D with positive dimensions
  NVTEShape logical_shape;

  NVTEGroupedTensor nvte_tensor;

  /*! \brief Whether scaling factors are in format expected by GEMM
   *
   *  Only meaningful for MXFP8 and NVFP4.
   */
  bool with_gemm_swizzled_scales = false;

  /*! Map from NVTEGroupedTensorParam to parameter sizes */
  static constexpr size_t attr_sizes[] = {
      sizeof(NVTEBasicTensor),  // kNVTEGroupedRowwiseData
      sizeof(NVTEBasicTensor),  // kNVTEGroupedColumnwiseData
      sizeof(NVTEBasicTensor),  // kNVTEGroupedScale
      sizeof(NVTEBasicTensor),  // kNVTEGroupedAmax
      sizeof(NVTEBasicTensor),  // kNVTEGroupedRowwiseScaleInv
      sizeof(NVTEBasicTensor),  // kNVTEGroupedColumnwiseScaleInv
      sizeof(NVTEBasicTensor),  // kNVTEGroupedColumnwiseAmax
      sizeof(NVTEBasicTensor),  // kNVTEGroupedFirstDims
      sizeof(NVTEBasicTensor),  // kNVTEGroupedLastDims
      sizeof(NVTEBasicTensor),  // kNVTEGroupedTensorOffsets
      sizeof(uint8_t)           // kNVTEGroupedWithGEMMSwizzledScales
  };

  GroupedTensor(NVTEScalingMode scaling_mode, size_t num_tensors)
      : data(),
        columnwise_data(),
        scale_inv(),
        columnwise_scale_inv(),
        amax(),
        columnwise_amax(),
        scale(),
        scaling_mode(scaling_mode),
        num_tensors(num_tensors),
        first_dims(nullptr, std::vector<size_t>{0}, DType::kInt64),
        last_dims(nullptr, std::vector<size_t>{0}, DType::kInt64),
        tensor_offsets(nullptr, std::vector<size_t>{0}, DType::kInt64),
        logical_shape(nvte_make_shape(nullptr, 1)),
        nvte_tensor(0) {}

  explicit operator NVTEGroupedTensor() const noexcept { return nvte_tensor; }

  bool has_data() const noexcept { return data.has_data(); }
  bool has_columnwise_data() const noexcept { return columnwise_data.has_data(); }

  bool all_same_first_dim() const noexcept { return !first_dims.has_data(); }
  bool all_same_last_dim() const noexcept { return !last_dims.has_data(); }
  bool all_same_shape() const noexcept { return !first_dims.has_data() && !last_dims.has_data(); }
  bool varying_both_dims() const noexcept { return first_dims.has_data() && last_dims.has_data(); }

  size_t get_common_first_dim() const {
    NVTE_CHECK(all_same_first_dim(), "First dim varies across tensors");
    NVTE_CHECK(logical_shape.ndim == 2, "Logical shape must be 2D");
    if (all_same_shape()) {
      // When both dims are uniform: logical_shape = [num_tensors * M, N]
      return logical_shape.data[0] / num_tensors;
    } else {
      // When varying last dims but not first dim: logical_shape = [M, sum_of_last_dims]
      return logical_shape.data[0];
    }
  }
  size_t get_common_last_dim() const {
    NVTE_CHECK(all_same_last_dim(), "Last dim varies across tensors");
    NVTE_CHECK(logical_shape.ndim == 2, "Logical shape must be 2D");
    // For both uniform and varying first dim cases: logical_shape[1] is the common last dim
    return logical_shape.data[1];
  }

  DType dtype() const {
    if (!has_data() && has_columnwise_data()) {
      return columnwise_data.dtype;
    }
    return data.dtype;
  }

  void clear() {
    data.clear();
    columnwise_data.clear();
    scale_inv.clear();
    columnwise_scale_inv.clear();
    amax.clear();
    columnwise_amax.clear();
    scale.clear();
    first_dims.clear();
    last_dims.clear();
    tensor_offsets.clear();
    logical_shape = nvte_make_shape(nullptr, 1);
    num_tensors = 0;
    scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
    nvte_tensor = 0;
    with_gemm_swizzled_scales = false;
  }
};


// GroupedTensor allocator - similar pattern to TensorAllocator
class GroupedTensorAllocator {
 public:
  static GroupedTensorAllocator &instance() {
    static GroupedTensorAllocator allocator;
    return allocator;
  }

  ~GroupedTensorAllocator() {}

  NVTEGroupedTensor Allocate(NVTEScalingMode mode, size_t num_tensors, NVTEShape logical_shape) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!free_list.empty()) {
      uintptr_t index = free_list.back();
      NVTEGroupedTensor ret = reinterpret_cast<NVTEGroupedTensor>(index);
      free_list.pop_back();
      // 1-based indexing - fully reinitialize the tensor to avoid stale data
      memory[index - 1].scaling_mode = mode;
      memory[index - 1].num_tensors = num_tensors;
      memory[index - 1].logical_shape = logical_shape;
      memory[index - 1].nvte_tensor = ret;
      return ret;
    }
    if (memory.size() < memory.capacity()) {
      memory.emplace_back(mode, num_tensors);
      GroupedTensor &t = memory.back();
      size = memory.size();
      // 1-based indexing
      uintptr_t index = memory.size();
      t.logical_shape = logical_shape;
      t.nvte_tensor = reinterpret_cast<NVTEGroupedTensor>(index);
      return reinterpret_cast<NVTEGroupedTensor>(index);
    }
    NVTE_ERROR(
        "Cannot allocate a new NVTEGroupedTensor. Maximum number of grouped tensors reached: ",
        MAX_GROUPED_TENSOR_NUM, ". There is probably a memory leak in your application.");
  }

  void Free(NVTEGroupedTensor t) {
    uintptr_t index = reinterpret_cast<uintptr_t>(t);
    if (index == 0) return;
    std::lock_guard<std::mutex> lock(mutex);
    NVTE_CHECK(index <= memory.size(), "Invalid grouped tensor.");
    free_list.push_back(index);
    // Clean up
    memory[index - 1].clear();
  }

  GroupedTensor *convertNVTEGroupedTensor(NVTEGroupedTensor t) {
    uintptr_t index = reinterpret_cast<uintptr_t>(t);
    // 1-based indexing to enable 0-initialization of NVTEGroupedTensor
    // to be invalid tensor
    static_assert(nullptr == 0);
    if (index != 0 && index <= size) {
      return &(memory[index - 1]);
    }
    return nullptr;
  }

 private:
  GroupedTensorAllocator() {
    std::lock_guard<std::mutex> lock(mutex);
    memory.reserve(MAX_GROUPED_TENSOR_NUM);
  }

  std::mutex mutex;
  std::atomic<size_t> size;
  // Allocate at most 20 MB for grouped tensors
  const size_t MAX_GROUPED_TENSOR_NUM = 20 * 1024 * 1024 / sizeof(GroupedTensor);
  std::vector<uintptr_t> free_list;
  std::vector<GroupedTensor> memory;
};


class TensorAllocator {
 public:
  static TensorAllocator &instance() {
    static TensorAllocator allocator;
    return allocator;
  }

  ~TensorAllocator() {}

  NVTETensor Allocate(NVTEScalingMode mode) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!free_list.empty()) {
      uintptr_t index = free_list.back();
      NVTETensor ret = reinterpret_cast<NVTETensor>(index);
      free_list.pop_back();
      if (debug) {
        std::cout << "Allocated " << index
                  << " from free list. Free list size: " << free_list.size() << " and capacity "
                  << free_list.capacity() << std::endl;
      }
      // 1-based indexing
      memory[index - 1].scaling_mode = mode;
      return ret;
    }
    if (memory.size() < memory.capacity()) {
      memory.emplace_back();
      Tensor &t = memory.back();
      size = memory.size();
      // 1-based indexing
      uintptr_t index = memory.size();
      if (debug) {
        std::cout << "Allocated " << index << ". Memory size: " << memory.size() << " and capacity "
                  << memory.capacity() << std::endl;
      }
      t.scaling_mode = mode;
      t.nvte_tensor = reinterpret_cast<NVTETensor>(index);
      return reinterpret_cast<NVTETensor>(index);
    }
    NVTE_ERROR("Cannot allocate a new NVTETensor. Maximum number of tensors reached: ",
               MAX_TENSOR_NUM, ". There is probably a memory leak in your application.");
  }

  void Free(NVTETensor t) {
    uintptr_t index = reinterpret_cast<uintptr_t>(t);
    if (index == 0) return;
    std::lock_guard<std::mutex> lock(mutex);
    NVTE_CHECK(index <= memory.size(), "Invalid tensor.");
    free_list.push_back(index);
    // Clean up
    memory[index - 1].clear();
    if (debug) {
      std::cout << "Freed " << index << ". Free list size: " << free_list.size() << " and capacity "
                << free_list.capacity() << std::endl;
    }
  }

  void Free(NVTETensor *t, size_t N) {
    std::lock_guard<std::mutex> lock(mutex);
    for (size_t i = 0; i < N; ++i) {
      uintptr_t index = reinterpret_cast<uintptr_t>(t[i]);
      if (index == 0) continue;
      NVTE_CHECK(index <= memory.size(), "Invalid tensor.");
      free_list.push_back(index);
      // Clean up
      memory[index - 1].clear();
    }
    if (debug) {
      std::cout << "Freed range of" << N << " tensors. Free list size: " << free_list.size()
                << " and capacity " << free_list.capacity() << std::endl;
    }
  }

  Tensor *convertNVTETensor(NVTETensor t) {
    uintptr_t index = reinterpret_cast<uintptr_t>(t);
    // 1-based indexing to enable 0-initialization of NVTETensor
    // to be invalid tensor
    static_assert(nullptr == 0);
    if (index != 0 && index <= size) {
      return &(memory[index - 1]);
    }
    return nullptr;
  }

  void setDebug(bool debug) {
    std::lock_guard<std::mutex> lock(mutex);
    this->debug = debug;
  }

 private:
  TensorAllocator() {
    std::lock_guard<std::mutex> lock(mutex);
    memory.reserve(MAX_TENSOR_NUM);
  }

  std::mutex mutex;
  std::atomic<size_t> size;
  // Allocate at most 20 MB for tensors
  // Should be replaced by virtual memory allocation
  const size_t MAX_TENSOR_NUM = 20 * 1024 * 1024 / sizeof(Tensor);
  std::vector<uintptr_t> free_list;
  std::vector<Tensor> memory;
  bool debug = false;
};

	
inline Tensor *convertNVTETensor(const NVTETensor t) {
  return reinterpret_cast<Tensor *>(t);
}

inline Tensor *convertNVTETensorCheck(const NVTETensor t) {
  Tensor *ptr = convertNVTETensor(t);
  NVTE_CHECK(ptr != nullptr, "Invalid tensor.");
  return ptr;
}



constexpr size_t kThreadsPerWarp = 32;
constexpr int kThreadsPerBlock =
    128;  // Using 4 warps in 1 CTA, Each warp is responsible for 1 token.
constexpr float epsilon = 1e-20;

template <typename T>
__device__ inline T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__device__ inline T sum(T a, T b) {
  return a + b;
}

enum ReduceFuncType {
  SUM,
  MAX,
};

template <typename T>
__device__ inline T warp_reduce_on_shmem(T *data_ptr, int data_size, ReduceFuncType type,
                                         int lane_id) {
  T (*reduce_func)(T, T);
  double default_val = 0;
  if (type == ReduceFuncType::SUM) {
    reduce_func = sum;
    default_val = 0;
  } else if (type == ReduceFuncType::MAX) {
    reduce_func = max;
    default_val = -std::numeric_limits<double>::infinity();
  }

  // Some value is hanlded in local thread
  // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
  // Reduce the value in local thread
  volatile double val = lane_id < data_size ? static_cast<double>(data_ptr[lane_id]) : default_val;
  for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
    val = reduce_func(val, data_ptr[i]);
  }

  // Warp shuffle between threads
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 1));
  __syncwarp();
  return T(val);
}

template <typename DataType>
__device__ inline void apply_sigmoid_on_float(DataType *scores, int data_size, int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = static_cast<float>(1.0f / (1.0f + exp(-static_cast<float>(scores[i]))));
  }
}

template <typename T>
__device__ inline T masked_warp_reduce_on_shmem(T *data_ptr, bool *mask, int data_size,
                                                ReduceFuncType type, int lane_id) {
  T (*reduce_func)(T, T);
  double default_val = 0;
  if (type == ReduceFuncType::SUM) {
    reduce_func = sum;
    default_val = 0;
  } else if (type == ReduceFuncType::MAX) {
    reduce_func = max;
    default_val = -std::numeric_limits<double>::infinity();
  }

  // Some value is hanlded in local thread
  // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
  // Reduce the value in local thread
  volatile double val =
      lane_id < data_size && mask[lane_id] ? static_cast<double>(data_ptr[lane_id]) : default_val;
  for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
    if (mask[i]) {
      val = reduce_func(val, data_ptr[i]);
    }
  }

  // Warp shuffle between threads
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 1));
  __syncwarp();
  return T(val);
}

template <typename DataType>
__device__ inline void apply_sigmoid_bwd_on_float(DataType *grad, DataType *fwd_output,
                                                  int data_size, int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    grad[i] = static_cast<double>(grad[i]) * static_cast<double>(fwd_output[i]) *
              (1 - static_cast<double>(fwd_output[i]));
  }
}

template <typename DataType>
__device__ inline void apply_softmax_bwd_on_float(DataType *grad, DataType *fwd_output,
                                                  DataType *comp_buf, bool *mask, int data_size,
                                                  int lane_id) {
  // Put the result of output * grad to the comp_buf
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    if (mask) {
      if (mask[i])
        comp_buf[i] = static_cast<float>(grad[i]) * static_cast<float>(fwd_output[i]);
      else
        comp_buf[i] = 0.0f;
    } else {
      comp_buf[i] = static_cast<float>(grad[i]) * static_cast<float>(fwd_output[i]);
    }
  }
  __syncwarp();
  float sum_Output_x_Grad = warp_reduce_on_shmem(
      /*data ptr = */ comp_buf,
      /*data size = */ data_size,
      /*reduce func = */ ReduceFuncType::SUM, lane_id);
  // In-place update
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    if (mask) {
      if (mask[i])
        grad[i] =
            static_cast<float>(fwd_output[i]) * (static_cast<float>(grad[i]) - sum_Output_x_Grad);
      else
        grad[i] = 0.0f;
    } else {
      grad[i] =
          static_cast<float>(fwd_output[i]) * (static_cast<float>(grad[i]) - sum_Output_x_Grad);
    }
  }
}

template <typename DataType>
__device__ inline void apply_softmax_on_float(DataType *scores, int data_size, int lane_id) {
  // 1. compute the max of value
  float max_val =
      static_cast<float>(warp_reduce_on_shmem(scores, data_size, ReduceFuncType::MAX, lane_id));
  // 2. value -> exp_value
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = static_cast<float>(exp(static_cast<float>(scores[i]) - max_val));
  }
  __syncwarp();
  // 3. compute the sum of exp_value
  float sum_val =
      static_cast<float>(warp_reduce_on_shmem(scores, data_size, ReduceFuncType::SUM, lane_id));
  // 4. update the softmax value
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = static_cast<float>(scores[i]) / sum_val;
  }
  __syncwarp();
}

template <typename T>
__device__ inline void naive_topk_and_mask(T *scores, int data_size, int topk, int *topk_indices,
                                           T *topk_scores, int lane_id) {
  // Check if the index is masked by the later iteration
  auto is_masked = [&topk_indices](int k, int index) {
    if (k == 0) return false;
    for (int i = 0; i < k; i++) {
      if (topk_indices[i] == index) return true;
    }
    return false;
  };
  // Topk Times: Find the max value and its index
  // Then mask it, and record the index in the topk_indices
  // After looping topk times, the topk_indices will be the topk indices
  for (int k = 0; k < topk; k++) {
    // Find the max value and its index
    volatile double val = (lane_id < data_size && !is_masked(k, lane_id))
                              ? static_cast<double>(scores[lane_id])
                              : -std::numeric_limits<double>::infinity();
    volatile int index = (lane_id < data_size) ? lane_id : 0;
    // Some value is hanlded in local thread
    // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
    // Reduce the value in local thread
    for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
      volatile double cur_val = (is_masked(k, i)) ? -std::numeric_limits<double>::infinity()
                                                  : static_cast<double>(scores[i]);
      if (cur_val > val) {
        val = cur_val;
        index = i;
      }
    }
    // Warp shuffle between threads
    for (int s = 16; s > 0; s /= 2) {
      volatile auto shuffled_val = __shfl_xor_sync(0xffffffff, val, s);
      volatile auto shuffled_index = __shfl_xor_sync(0xffffffff, index, s);
      if (shuffled_val > val) {
        val = shuffled_val;
        index = shuffled_index;
      }
    }
    if (lane_id == 0) {
      topk_indices[k] = index;
      topk_scores[k] = val;
    }
    __syncwarp();
  }
}

// Current TE only support float32/bf16/fp16, float64 probs should be considered in the future
#define TE_ROUTER_PROBS_TYPE_SWITCH_ALL(dtype, type, ...) \
  switch (dtype) {                                        \
    using namespace transformer_engine;                   \
    case DType::kFloat32: {                               \
      using type = float;                                 \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kFloat16: {                               \
      using type = fp16;                                  \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kBFloat16: {                              \
      using type = bf16;                                  \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    default:                                              \
      NVTE_ERROR("Invalid type.");                        \
  }

#define TE_ROUTER_INDEX_TYPE_SWITCH_ALL(dtype, type, ...) \
  switch (dtype) {                                        \
    using namespace transformer_engine;                   \
    case DType::kInt32: {                                 \
      using type = int32_t;                               \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kInt64: {                                 \
      using type = int64_t;                               \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kBFloat16: {                              \
      using type = bf16;                                  \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kFloat32: {                               \
      using type = float;                                 \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    default:                                              \
      NVTE_ERROR("Invalid type.");                        \
  }
}  // namespace transformer_engine
#endif

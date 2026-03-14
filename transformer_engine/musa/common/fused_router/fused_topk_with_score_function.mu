/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <musa_runtime.h>
#include <transformer_engine/fused_router.h>
#include <type_traits>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.muh"
#include "utils.h"

/*
Tensor *convertNVTETensorCheck(const NVTETensor t) {
  Tensor *ptr = TensorAllocator::instance().convertNVTETensor(t);
  NVTE_CHECK(ptr != nullptr, "Invalid tensor.");
  return ptr;
}
*/

namespace transformer_engine {

template <typename DataType, typename BiasType, int score_function=0, bool use_pre_softmax=false,
          int kNumExpertsConst=0, int kTopKConst=0, int kNumGroupsConst=0, int kGroupTopKConst=0>
__global__ void fused_topk_with_score_function_forward_kernel(
    const DataType *logits, int num_tokens, int num_experts, int topk,
    int num_groups, int group_topk, float scaling_factor,
    const BiasType *expert_bias, DataType *probs, bool *routing_map,
    DataType *intermediate_output) {
  constexpr bool kFixedNumExperts = (kNumExpertsConst > 0);
  constexpr bool kFixedTopK = (kTopKConst > 0);
  constexpr bool kFixedNumGroups = (kNumGroupsConst > 0);
  constexpr bool kFixedGroupTopK = (kGroupTopKConst > 0);
  const int num_experts_v = kFixedNumExperts ? kNumExpertsConst : num_experts;
  const int topk_v = kFixedTopK ? kTopKConst : topk;
  const int num_groups_v = kFixedNumGroups ? kNumGroupsConst : num_groups;
  const int group_topk_v = kFixedGroupTopK ? kGroupTopKConst : group_topk;
  const int topk_per_group_v = group_topk_v > 0 ? (topk_v / group_topk_v) : 0;

  /***
     * Section: Global Variables/Addresses init
     * - Assume the sizeof(DataType) >= sizeof(int),
     *   So DataType address is assigned firstly to avoid the alignment issue
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  // Used variables/addresses init
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];
  DataType *scores_buf = reinterpret_cast<DataType *>(shmem);
  DataType *topk_scores_buf =
      reinterpret_cast<DataType *>(scores_buf + num_experts_v * num_token_per_block);
  DataType *group_scores_buf = nullptr, *masked_scores_buf = nullptr;
  int *topk_indices_buf = nullptr;
  if (group_topk_v > 0) {
    masked_scores_buf = reinterpret_cast<DataType *>(topk_scores_buf + topk_v * num_token_per_block);
    group_scores_buf =
        reinterpret_cast<DataType *>(masked_scores_buf + num_experts_v * num_token_per_block);
    topk_indices_buf = reinterpret_cast<int *>(group_scores_buf + num_groups_v * num_token_per_block);
  } else {
    topk_indices_buf = reinterpret_cast<int *>(topk_scores_buf + topk_v * num_token_per_block);
  }
  // The address of buffers on the current warp
  DataType *scores = scores_buf + warp_id * num_experts_v;
  DataType *topk_scores = topk_scores_buf + warp_id * topk_v;
  DataType *masked_scores =
      (group_topk_v > 0) ? (masked_scores_buf + warp_id * num_experts_v) : nullptr;
  DataType *group_scores =
      (group_topk_v > 0) ? (group_scores_buf + warp_id * num_groups_v) : nullptr;
  int *topk_indices = topk_indices_buf + warp_id * topk_v;

  /***
     * Section: Main Loop
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    // Each warp is responsible for one token
    if (token_offset_cur_warp >= num_tokens) break;

    /***
         * Section: Init buffer
         * - Clear the global buffer which will accept the result of this round
         * - Clear/Init the shmem buffer used by current warp this round
         * - Load the logits to shmem
         */
    int pos_offset = token_offset_cur_warp * num_experts_v;
    float pre_softmax_max = -std::numeric_limits<float>::infinity();
    // Load logits to shared memory, and fuse local max reduction for pre-softmax path.
    if constexpr (std::is_same<DataType, float>::value) {
      if ((num_experts_v & 1) == 0) {
        const float2 *logits_vec = reinterpret_cast<const float2 *>(logits + pos_offset);
        float2 *scores_vec = reinterpret_cast<float2 *>(scores);
#pragma unroll
        for (int i = lane_id; i < (num_experts_v / 2); i += kThreadsPerWarp) {
          float2 v = logits_vec[i];
          scores_vec[i] = v;
          if constexpr (use_pre_softmax && score_function == 1) {
            pre_softmax_max = v.x > pre_softmax_max ? v.x : pre_softmax_max;
            pre_softmax_max = v.y > pre_softmax_max ? v.y : pre_softmax_max;
          }
        }
      } else {
        for (int i = lane_id; i < num_experts_v; i += kThreadsPerWarp) {
          DataType v = logits[pos_offset + i];
          scores[i] = v;
          if constexpr (use_pre_softmax && score_function == 1) {
            float fv = static_cast<float>(v);
            pre_softmax_max = fv > pre_softmax_max ? fv : pre_softmax_max;
          }
        }
      }
    } else {
      for (int i = lane_id; i < num_experts_v; i += kThreadsPerWarp) {
        DataType v = logits[pos_offset + i];
        scores[i] = v;
        if constexpr (use_pre_softmax && score_function == 1) {
          float fv = static_cast<float>(v);
          pre_softmax_max = fv > pre_softmax_max ? fv : pre_softmax_max;
        }
      }
    }
    if constexpr (use_pre_softmax && score_function == 1) {
      for (int offset = kThreadsPerWarp / 2; offset > 0; offset /= 2) {
        float shuffled_max = __shfl_down_sync(0xffffffff, pre_softmax_max, offset);
        pre_softmax_max = shuffled_max > pre_softmax_max ? shuffled_max : pre_softmax_max;
      }
      pre_softmax_max = __shfl_sync(0xffffffff, pre_softmax_max, 0);
    }
    // If group_topk > 0, init the masked_scores to -inf
    if (group_topk_v > 0) {
      for (int i = lane_id; i < num_experts_v; i += kThreadsPerWarp) {
        masked_scores[i] = -std::numeric_limits<DataType>::infinity();
      }
    }
    __syncwarp();

    /***
         * Section: Preprocess
         * Possible preprocess the scores before the topk operation
         * - Pre-softmax
         * - Sigmoid
         * - Expert bias
         * This is in-place scores update
         */
    // score_function == 1 means softmax
    if constexpr (use_pre_softmax && score_function == 1) {
      // Apply softmax to the logits before the topk
      apply_softmax_on_float_with_max(scores, num_experts_v, lane_id, pre_softmax_max);
      __syncwarp();
      // Save the softmax output for backward
      for (int i = lane_id; i < num_experts_v; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
    }

    // score_function == 0 means sigmoid
    if constexpr (score_function == 0) {
      // Apply sigmoid to the logits
      apply_sigmoid_on_float(scores, num_experts_v, lane_id);
      __syncwarp();
      // Save the sigmoid output for backward
      for (int i = lane_id; i < num_experts_v; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
    }

    __syncwarp();  //Confirm the scores is written to the softmax/sigmoid output

    // Expert bias is only used at the sigmoid case
    if (expert_bias && score_function == 0) {
      for (int i = lane_id; i < num_experts_v; i += kThreadsPerWarp) {
        scores[i] = static_cast<DataType>(static_cast<double>(scores[i]) +
                                          static_cast<double>(expert_bias[i]));
      }
    }
    __syncwarp();

    /***
         * Section: Topk
         * Get the topk indices
         * - group_topk
         * - naive topk
         * - topk with expert bias
         */
    // Topk on the scores
    // The bias is not empty only happens at the sigmod case
    if (group_topk_v > 0) {
      int group_size = num_experts_v / num_groups_v;
      // Top2
      for (int i = 0; i < num_groups_v; i++) {
        if constexpr (kFixedTopK && kFixedGroupTopK) {
          constexpr int kFixedGroupInnerTopK = kTopKConst / kGroupTopKConst;
          naive_topk_and_mask_constexpr<DataType, kFixedGroupInnerTopK>(
              /*scores ptr = */ scores + i * group_size,
              /*data size = */ group_size,
              /*topk indices ptr = */ topk_indices,
              /*topk scores ptr = */ topk_scores,
              /*lane id = */ lane_id);
        } else {
          naive_topk_and_mask(
              /*scores ptr = */ scores + i * group_size,
              /*data size = */ group_size,
              /*topk = */ topk_per_group_v,
              /*topk indices ptr = */ topk_indices,
              /*topk scores ptr = */ topk_scores,
              /*lane id = */ lane_id);
        }
        __syncwarp();
        // Compute the group score
        if (lane_id == 0) {
          DataType tmp = 0.0f;
          if constexpr (kFixedTopK && kFixedGroupTopK) {
#pragma unroll
            for (int j = 0; j < (kTopKConst / kGroupTopKConst); j++) {
              tmp = tmp + topk_scores[j];
            }
          } else {
            for (int j = 0; j < topk_per_group_v; j++) {
              tmp = tmp + topk_scores[j];
            }
          }
          group_scores[i] = tmp;
        }
        __syncwarp();
      }

      // select the topk groups
      if constexpr (kFixedGroupTopK) {
        naive_topk_and_mask_constexpr<DataType, kGroupTopKConst>(
            /*scores ptr = */ group_scores,
            /*data size = */ num_groups_v,
            /*topk indices ptr = */ topk_indices,
            /*topk scores ptr = */ topk_scores,
            /*lane id = */ lane_id);
      } else {
        naive_topk_and_mask(
            /*scores ptr = */ group_scores,
            /*data size = */ num_groups_v,
            /*topk = */ group_topk_v,
            /*topk indices ptr = */ topk_indices,
            /*topk scores ptr = */ topk_scores,
            /*lane id = */ lane_id);
      }
      __syncwarp();
      // Copy the unmasked scores to the buffer
      for (int i = 0; i < group_topk_v; i++) {
        int st = topk_indices[i] * group_size;
        int ed = st + group_size;
        for (int j = st + lane_id; j < ed; j += kThreadsPerWarp) {
          masked_scores[j] = scores[j];
        }
      }
      __syncwarp();
      if constexpr (kFixedTopK) {
        naive_topk_and_mask_constexpr<DataType, kTopKConst>(
            masked_scores, num_experts_v, topk_indices, topk_scores, lane_id);
      } else {
        naive_topk_and_mask(
            masked_scores, num_experts_v, topk_v, topk_indices, topk_scores, lane_id);
      }

    } else {
      if constexpr (kFixedTopK) {
        naive_topk_and_mask_constexpr<DataType, kTopKConst>(
            scores, num_experts_v, topk_indices, topk_scores, lane_id);
      } else {
        naive_topk_and_mask(scores, num_experts_v, topk_v, topk_indices, topk_scores, lane_id);
      }
    }
    __syncwarp();

    /***
         * Section: Postprocess
         * Possible postprocess the scores after the topk operation
         * - Revert Expert bias
         * - Softmax
         * - Sigmoid post-processing when topk > 1
         * - Write the result with scaling_factor
         */
    // Revert Expert bias from the topk scores
    if (expert_bias && score_function == 0) {
      for (int i = lane_id; i < topk_v; i += kThreadsPerWarp) {
        topk_scores[i] =
            static_cast<double>(topk_scores[i]) - static_cast<double>(expert_bias[topk_indices[i]]);
      }
    }
    __syncwarp();

    // score_function == 1 means softmax
    if constexpr (!use_pre_softmax && score_function == 1) {
      // Apply softmax to the topk logits
      apply_softmax_on_float(topk_scores, topk_v, lane_id);
      __syncwarp();
      // Save the softmax output for backward
      for (int i = lane_id; i < topk_v; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + topk_indices[i]] = topk_scores[i];
      }
    }

    // score_function == 0 means sigmoid
    if constexpr (score_function == 0) {
      if (topk_v > 1) {
        double sum_scores =
            warp_reduce_on_shmem(topk_scores, topk_v, ReduceFuncType::SUM, lane_id);
        for (int i = lane_id; i < topk_v; i += kThreadsPerWarp) {
          topk_scores[i] = static_cast<double>(topk_scores[i]) / (sum_scores + epsilon);
        }
      }
      __syncwarp();
    }

    // Write the probs/routing_map to the output tensor
    for (int i = lane_id; i < topk_v; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
      probs[pos_offset + topk_indices[i]] = scaling_factor * static_cast<double>(topk_scores[i]);
    }
    __syncwarp();
  }
}

template <typename DataType, typename BiasType>
void fused_topk_with_score_function_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const BiasType *expert_bias, DataType *probs, bool *routing_map, DataType *intermediate_output,
    musaStream_t stream) {
  // Zero init output tensors once before kernel launch.
  size_t output_elems = static_cast<size_t>(num_tokens) * static_cast<size_t>(num_experts);
  NVTE_CHECK_CUDA(musaMemsetAsync(probs, 0, output_elems * sizeof(DataType), stream));
  NVTE_CHECK_CUDA(musaMemsetAsync(routing_map, 0, output_elems * sizeof(bool), stream));

  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = ((num_tokens + num_token_per_block - 1) / num_token_per_block);
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(DataType)  // scores
                              + topk * num_token_per_block * sizeof(DataType)       // topk_scores
                              + topk * num_token_per_block * sizeof(int);           // topk_indices
  if (group_topk > 0) {
    shared_memory_size += num_groups * num_token_per_block * sizeof(DataType);   // group_scores
    shared_memory_size += num_experts * num_token_per_block * sizeof(DataType);  // maksed_scores
  }

#define TE_LAUNCH_FUSED_TOPK_FWD(NUM_EXPERTS_C, TOPK_C, NUM_GROUPS_C, GROUP_TOPK_C)                \
  do {                                                                                               \
    if (score_function == 0 && use_pre_softmax == false) {                                           \
      fused_topk_with_score_function_forward_kernel<DataType, BiasType, 0, false, NUM_EXPERTS_C,    \
                                                    TOPK_C, NUM_GROUPS_C, GROUP_TOPK_C>              \
          <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(                             \
              logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,        \
              expert_bias, probs, routing_map, intermediate_output);                                 \
    } else if (score_function == 0 && use_pre_softmax == true) {                                     \
      fused_topk_with_score_function_forward_kernel<DataType, BiasType, 0, true, NUM_EXPERTS_C,     \
                                                    TOPK_C, NUM_GROUPS_C, GROUP_TOPK_C>              \
          <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(                             \
              logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,        \
              expert_bias, probs, routing_map, intermediate_output);                                 \
    } else if (score_function == 1 && use_pre_softmax == false) {                                    \
      fused_topk_with_score_function_forward_kernel<DataType, BiasType, 1, false, NUM_EXPERTS_C,    \
                                                    TOPK_C, NUM_GROUPS_C, GROUP_TOPK_C>              \
          <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(                             \
              logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,        \
              expert_bias, probs, routing_map, intermediate_output);                                 \
    } else if (score_function == 1 && use_pre_softmax == true) {                                     \
      fused_topk_with_score_function_forward_kernel<DataType, BiasType, 1, true, NUM_EXPERTS_C,     \
                                                    TOPK_C, NUM_GROUPS_C, GROUP_TOPK_C>              \
          <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(                             \
              logits, num_tokens, num_experts, topk, num_groups, group_topk, scaling_factor,        \
              expert_bias, probs, routing_map, intermediate_output);                                 \
    } else {                                                                                          \
      assert(false && "Invalid combination of score_function and use_pre_softmax");                  \
    }                                                                                                 \
  } while (0)

  bool launched_specialized = false;
  if (topk == 8) {
    if (group_topk <= 0) {
      if (num_experts == 128) {
        TE_LAUNCH_FUSED_TOPK_FWD(128, 8, 0, 0);
        launched_specialized = true;
      } else if (num_experts == 192) {
        TE_LAUNCH_FUSED_TOPK_FWD(192, 8, 0, 0);
        launched_specialized = true;
      }
    } else if (num_groups == 8 && group_topk == 4) {
      if (num_experts == 128) {
        TE_LAUNCH_FUSED_TOPK_FWD(128, 8, 8, 4);
        launched_specialized = true;
      } else if (num_experts == 192) {
        TE_LAUNCH_FUSED_TOPK_FWD(192, 8, 8, 4);
        launched_specialized = true;
      }
    } else if (num_groups == 16 && group_topk == 2) {
      if (num_experts == 128) {
        TE_LAUNCH_FUSED_TOPK_FWD(128, 8, 16, 2);
        launched_specialized = true;
      } else if (num_experts == 192) {
        TE_LAUNCH_FUSED_TOPK_FWD(192, 8, 16, 2);
        launched_specialized = true;
      }
    }
  }

  if (!launched_specialized) {
    TE_LAUNCH_FUSED_TOPK_FWD(0, 0, 0, 0);
  }

#undef TE_LAUNCH_FUSED_TOPK_FWD
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_topk_with_score_function_forward(const Tensor logits, int num_tokens, int num_experts,
                                            int topk, bool use_pre_softmax, int num_groups,
                                            int group_topk, float scaling_factor,
                                            int score_function, const Tensor expert_bias,
                                            Tensor probs, Tensor routing_map,
                                            Tensor intermediate_output, musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      logits.data.dtype, DataType,
      if (expert_bias.has_data()) {
        TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
            expert_bias.data.dtype, BiasType,
            fused_topk_with_score_function_forward_kernel_launcher<DataType, BiasType>(
                reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,
                use_pre_softmax, num_groups, group_topk, scaling_factor, score_function,
                reinterpret_cast<BiasType *>(expert_bias.data.dptr),
                reinterpret_cast<DataType *>(probs.data.dptr),
                reinterpret_cast<bool *>(routing_map.data.dptr),
                reinterpret_cast<DataType *>(intermediate_output.data.dptr), stream););
      } else {
        fused_topk_with_score_function_forward_kernel_launcher<DataType, DataType>(
            reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,
            use_pre_softmax, num_groups, group_topk, scaling_factor, score_function, nullptr,
            reinterpret_cast<DataType *>(probs.data.dptr),
            reinterpret_cast<bool *>(routing_map.data.dptr),
            reinterpret_cast<DataType *>(intermediate_output.data.dptr), stream);
      });
}

template <typename DataType, int score_function=0, bool use_pre_softmax=false>
__global__ void fused_topk_with_score_function_backward_kernel(
    // Inputs tensor
    const bool *routing_map, const DataType *intermediate_output, const DataType *grad_probs,
    // Other parameters
    int num_tokens, int num_experts, int topk, float scaling_factor,
    // Output tensor
    DataType *grad_logits) {
  /***
     * Section: Global Variables/Addresses init
     * - Assume the sizeof(DataType) >= sizeof(int),
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  // Used variables/addresses init
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];
  DataType *grad_probs_buf = reinterpret_cast<DataType *>(shmem);
  // To store the output of softmax/sigmoid from the fwd
  DataType *act_from_fwd_buf =
      reinterpret_cast<DataType *>(grad_probs_buf + num_experts * num_token_per_block);
  DataType *comp_buf =
      reinterpret_cast<DataType *>(act_from_fwd_buf + num_experts * num_token_per_block);
  // To store the routing_map from the fwd
  bool *routing_map_buf = reinterpret_cast<bool *>(comp_buf + num_experts * num_token_per_block);
  // The address of buffers on the current warp
  DataType *local_grad = grad_probs_buf + warp_id * num_experts;
  DataType *local_act_from_fwd = act_from_fwd_buf + warp_id * num_experts;
  DataType *local_comp_buf = comp_buf + warp_id * num_experts;
  bool *local_routing_map = routing_map_buf + warp_id * num_experts;

  /***
     * Section: Main Loop
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    // Each warp is responsible for one token
    if (token_offset_cur_warp >= num_tokens) break;

    /***
         * Section: Init buffer
         * - Clear the global buffer which will accept the result of this round
         * - Clear/Init the shmem buffer used by current warp this round
         * - Load the dgrad/output_from_fwd to shmem
         */
    int pos_offset = token_offset_cur_warp * num_experts;
    // Load the dgrad/output_from_fwd to shmem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_grad[i] = grad_probs[pos_offset + i];
      local_act_from_fwd[i] = intermediate_output[pos_offset + i];
      local_routing_map[i] = routing_map[pos_offset + i];
    }
    __syncwarp();

    /***
         * Section: Backward of ops after the topk
         * - Backward of the used scaling_factor
         * - Sigmoid Post-processing bwd when topk > 1
         * - Softmax bwd if use_pre_softmax is false
         */
    // Backward of the used scaling_factor
    // In-place update
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (local_routing_map[i]) {
        local_grad[i] = static_cast<double>(local_grad[i]) * scaling_factor;
      }
    }
    __syncwarp();
    // Sigmoid Post-processing bwd when topk > 1
    if (topk > 1 && score_function == 0) {
      double sum_fwd_input = masked_warp_reduce_on_shmem(
          /*data ptr = */ local_act_from_fwd,
          /*mask ptr = */ local_routing_map,
          /*data size = */ num_experts,
          /*reduce func = */ ReduceFuncType::SUM, lane_id);
      // Put the result of output * grad to the comp_buf
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_comp_buf[i] = (local_routing_map[i] ? static_cast<double>(local_grad[i]) *
                                                        static_cast<double>(local_act_from_fwd[i])
                                                  : 0.0f);
      }
      __syncwarp();
      double sum_Output_x_Grad = masked_warp_reduce_on_shmem(
          /*data ptr = */ local_comp_buf,
          /*mask ptr = */ local_routing_map,
          /*data size = */ num_experts,
          /*reduce func = */ ReduceFuncType::SUM, lane_id);
      // In-place update
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        if (local_routing_map[i]) {
          local_grad[i] =
              static_cast<double>(local_grad[i]) / (sum_fwd_input + epsilon) -
              sum_Output_x_Grad / ((sum_fwd_input + epsilon) * (sum_fwd_input + epsilon));
        } else {
          local_grad[i] = 0.0f;
        }
      }
    }
    __syncwarp();
    // Softmax bwd if use_pre_softmax is false
    if (!use_pre_softmax && score_function == 1) {
      apply_softmax_bwd_on_float(local_grad, local_act_from_fwd, local_comp_buf, local_routing_map,
                                 num_experts, lane_id);
      __syncwarp();
    }

    /***
         * Section: Backward of topk
         * mask the unselected position in the grad
         */
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (!local_routing_map[i]) {
        local_grad[i] = 0.0f;
      }
    }
    __syncwarp();

    /***
         * Section: Backward of ops before the topk
         * - Pre-softmax bwd
         * - Sigmoid bwd
         * - Write the grad_logits to the global mem
         */
    // Pre-softmax bwd
    if constexpr (score_function == 1 && use_pre_softmax) {
      apply_softmax_bwd_on_float(local_grad, local_act_from_fwd, local_comp_buf, nullptr,
                                 num_experts, lane_id);
      __syncwarp();
    }
    // Sigmoid bwd
    if constexpr (score_function == 0) {
      apply_sigmoid_bwd_on_float(local_grad, local_act_from_fwd, num_experts, lane_id);
      __syncwarp();
    }
    // Write grad_logits to global memory.
    if constexpr (std::is_same<DataType, float>::value) {
      if ((num_experts & 1) == 0) {
        float2 *grad_logits_vec = reinterpret_cast<float2 *>(grad_logits + pos_offset);
        float2 *local_grad_vec = reinterpret_cast<float2 *>(local_grad);
#pragma unroll
        for (int i = lane_id; i < (num_experts / 2); i += kThreadsPerWarp) {
          grad_logits_vec[i] = local_grad_vec[i];
        }
      } else {
#pragma unroll
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          grad_logits[pos_offset + i] = local_grad[i];
        }
      }
    } else {
#pragma unroll
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        grad_logits[pos_offset + i] = local_grad[i];
      }
    }
    __syncwarp();
  }
}

template <typename DataType>
void fused_topk_with_score_function_backward_kernel_launcher(
    const bool *routing_map, const DataType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType *grad_logits, musaStream_t stream) {
  // Zero init output tensor once before kernel launch.
  size_t output_elems = static_cast<size_t>(num_tokens) * static_cast<size_t>(num_experts);
  NVTE_CHECK_CUDA(musaMemsetAsync(grad_logits, 0, output_elems * sizeof(DataType), stream));

  // Meta data for the kernel
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(DataType)  // grad_probs
                              +
                              num_experts * num_token_per_block * sizeof(DataType)  // act_from_fwd
                              + num_experts * num_token_per_block * sizeof(DataType)  // comp_buf
                              + num_experts * num_token_per_block * sizeof(bool);     // routing_map

  if (score_function == 0 && use_pre_softmax == false)
    fused_topk_with_score_function_backward_kernel<DataType, 0, false>
        <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
            routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
            scaling_factor, grad_logits);
  else if (score_function == 0 && use_pre_softmax == true)
    fused_topk_with_score_function_backward_kernel<DataType, 0, true>
        <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
            routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
            scaling_factor, grad_logits);
  else if (score_function == 1 && use_pre_softmax == false)
    fused_topk_with_score_function_backward_kernel<DataType, 1, false>
        <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
            routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
            scaling_factor, grad_logits);
  else if (score_function == 1 && use_pre_softmax == true)
    fused_topk_with_score_function_backward_kernel<DataType, 1, true>
        <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
            routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
            scaling_factor, grad_logits);
  else
    assert(false && "Invalid combination of score_function and use_pre_softmax");
  NVTE_CHECK_CUDA(musaGetLastError());
}

void fused_topk_with_score_function_backward(const Tensor &routing_map,
                                             const Tensor &intermediate_output,
                                             const Tensor &grad_probs, int num_tokens,
                                             int num_experts, int topk, bool use_pre_softmax,
                                             float scaling_factor, int score_function,
                                             Tensor &grad_logits, musaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_logits.data.dtype, DataType,
      fused_topk_with_score_function_backward_kernel_launcher<DataType>(
          reinterpret_cast<bool *>(routing_map.data.dptr),
          reinterpret_cast<DataType *>(intermediate_output.data.dptr),
          reinterpret_cast<DataType *>(grad_probs.data.dptr), num_tokens, num_experts, topk,
          use_pre_softmax, scaling_factor, score_function,
          reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
}

}  // namespace transformer_engine

void nvte_fused_topk_with_score_function_forward(
    const NVTETensor logits, int num_tokens, int num_experts, int topk, int use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const NVTETensor expert_bias, NVTETensor probs, NVTETensor routing_map,
    NVTETensor intermediate_output, musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_forward);
  using namespace transformer_engine;
  Tensor *logits_tensor = convertNVTETensor(logits);
  NVTE_CHECK(logits_tensor != nullptr, "Invalid logits tensor.");
  Tensor *expert_bias_tensor = convertNVTETensor(expert_bias);
  Tensor *probs_tensor = convertNVTETensor(probs);
  NVTE_CHECK(probs_tensor != nullptr, "Invalid probs tensor.");
  Tensor *routing_map_tensor = convertNVTETensor(routing_map);
  NVTE_CHECK(routing_map_tensor != nullptr, "Invalid routing_map tensor.");
  Tensor *intermediate_output_tensor = convertNVTETensor(intermediate_output);
  NVTE_CHECK(intermediate_output_tensor != nullptr, "Invalid intermediate_output tensor.");
  Tensor empty_expert_bias;
  fused_topk_with_score_function_forward(
      *logits_tensor, num_tokens, num_experts, topk,
      static_cast<bool>(use_pre_softmax), num_groups, group_topk, scaling_factor, score_function,
      expert_bias_tensor != nullptr ? *expert_bias_tensor : empty_expert_bias,
      *probs_tensor, *routing_map_tensor, *intermediate_output_tensor, stream);
}

void nvte_fused_topk_with_score_function_backward(const NVTETensor routing_map,
                                                  const NVTETensor intermediate_output,
                                                  const NVTETensor grad_probs, int num_tokens,
                                                  int num_experts, int topk, int use_pre_softmax,
                                                  float scaling_factor, int score_function,
                                                  NVTETensor grad_logits, musaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_backward);
  using namespace transformer_engine;
  fused_topk_with_score_function_backward(
      *convertNVTETensorCheck(routing_map), *convertNVTETensorCheck(intermediate_output),
      *convertNVTETensorCheck(grad_probs), num_tokens, num_experts, topk,
      static_cast<bool>(use_pre_softmax), scaling_factor, score_function,
      *convertNVTETensorCheck(grad_logits), stream);
}

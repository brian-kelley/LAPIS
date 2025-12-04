#ifndef KOKKOSDNN_HPP
#define KOKKOSDNN_HPP

#include <Kokkos_Core.hpp>

#ifdef KOKKOSDNN_HAVE_CUDNN
#include <cudnn_frontend.h>

namespace KokkosDNN
{
  template<typename Scalar>
  fe::DataType_t getScalarType_cuDNN() {
    if constexpr(std::is_same_v<Scalar, float>) {
      return fe::DataType_t::FLOAT;
    }
    else if constexpr(std::is_same_v<Scalar, Kokkos::half_t>) {
      return fe::DataType_t::HALF
    }
    else {
      static_assert(false, "Unknown scalar type");
    }
    //?? bfloat16, double?
    return fe::DataType_t::FLOAT;
  }

  using Node = std::shared_ptr<fe::graph::Tensor_attributes>;

  template<typename Scalar>
  struct Graph
  {
    std::shared_ptr<cudnnHandle_t> handle;
    // cuDNN graph
    std::shared_ptr<fe::graph::Graph> graph;
    bool graphIsBuilt;
    // Map of input tensors and their currently bound data
    // (The data pointer can change on each execute call)
    std::unordered_map<Node, void*> inputTensors;
    // Map of constant tensors and their bound data
    // (The data pointer cannot be changed)
    std::unordered_map<Node, void*> constantTensors;

    Graph() {
      handle = std::make_shared<cudnnHandle_t>();
      cudnnCreate(handle.get());
      graph = std::make_shared<fe::graph::Graph>();
      // Use same scalar type for all tensors
      auto cudnnScalar = getScalarType_cuDNN<Scalar>();
      graph->set_io_data_type(cudnnScalar);
      graph->set_compute_data_type(cudnnScalar);
      graph->set_intermediate_data_type(cudnnScalar);
      // Graph still has to be populated with operations
      graphIsBuilt = false;
    }

    template<typename InputView>
    Node addInput(const InputView& input) {
      std::vector<int64_t> dims;
      std::vector<int64_t> strides;
      for(int i = 0; i < InputView::rank; i++) {
        dims.push_back(input.extent(0));
        strides.push_back(input.stride(0));
      }
      Node tensor = graph->tensor(
          fe::graph::Tensor_attributes()
          .set_name(input.label())
          .set_dim(dims)
          .set_stride(strides)
          .set_data_type(getScalarType_cuDNN<typename InputView::non_const_value_type>());
      inputTensors[tensor] = input.data();
      return tensor;
    }

    template<typename InputView>
    Node addConstant(const InputView& input) {
      std::vector<int64_t> dims;
      std::vector<int64_t> strides;
      for(int i = 0; i < InputView::rank; i++) {
        dims.push_back(input.extent(0));
        strides.push_back(input.stride(0));
      }
      Node tensor = graph->tensor(
          fe::graph::Tensor_attributes()
          .set_name(input.label())
          .set_dim(dims)
          .set_stride(strides)
          .set_data_type(getScalarType_cuDNN<typename InputView::non_const_value_type>());
      constantTensors[tensor] = input.data();
      return tensor;
    }

    void build() {
      REQUIRE(graph->build_operation_graph(handle).is_good());
      REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
      REQUIRE(graph->check_support().is_good());
      REQUIRE(graph->build_plans().is_good());
    }
  };
}

#endif

#endif // KOKKOSDNN_HPP

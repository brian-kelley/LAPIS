#ifndef KOKKOSDNN_HPP
#define KOKKOSDNN_HPP

#include <Kokkos_Core.hpp>

#ifdef KOKKOSDNN_HAVE_CUDNN
#include <cudnn_frontend.h>

#define KKDNN_ASSERT(val)
  if(!(val)) throw std::runtime_error("KokkosDNN operation failed: " #val)

namespace KokkosDNN
{
  template<typename Scalar>
  constexpr fe::DataType_t getScalarType_cuDNN() {
    if constexpr(std::is_same_v<Scalar, double>) {
      return fe::DataType_t::DOUBLE;
    }
    else if constexpr(std::is_same_v<Scalar, float>) {
      return fe::DataType_t::FLOAT;
    }
    else if constexpr(std::is_same_v<Scalar, Kokkos::Experimental::half_t>) {
      return fe::DataType_t::HALF;
    }
    else if constexpr(std::is_same_v<Scalar, Kokkos::Experimental::bhalf_t>) {
      return fe::DataType_t::BFLOAT16;
    }
    else {
      static_assert(false, "Unknown scalar type");
    }
    // ???
    return fe::DataType_t::FLOAT;
  }

  using Node = std::shared_ptr<fe::graph::Tensor_attributes>;
  using Workspace = Kokkos::View<int8_t*, Kokkos::CudaSpace>;

  template<typename Scalar>
  struct Graph
  {
    std::shared_ptr<cudnnHandle_t> handle;
    // cuDNN graph
    std::shared_ptr<fe::graph::Graph> graph;
    bool graphIsBuilt;
    // Map of constant tensors and their bound data
    // (The data pointer cannot be changed)
    std::unordered_map<Node, void*> constantTensors;
    // List of input tensors.
    std::vector<Node> inputTensors;
    std::vector<void*> boundInputData;
    // List of output tensors.
    std::vector<Node> outputTensors;
    std::vector<void*> boundOutputData;
    // Workspace view for intermediate tensors
    Workspace workspace;

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

    ~Graph() {
      cudnnDestroy(*handle);
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
      inputTensors.push_back(tensor)
      boundInputData.push_back(nullptr);
      return tensor;
    }

    template<typename ConstantView>
    Node addConstant(const ConstantView& view) {
      std::vector<int64_t> dims;
      std::vector<int64_t> strides;
      for(int i = 0; i < ConstantView::rank; i++) {
        dims.push_back(view.extent(0));
        strides.push_back(view.stride(0));
      }
      Node tensor = graph->tensor(
          fe::graph::Tensor_attributes()
          .set_name(view.label())
          .set_dim(dims)
          .set_stride(strides)
          .set_data_type(getScalarType_cuDNN<typename ConstantView::non_const_value_type>());
      constantTensors[tensor] = view.data();
      return tensor;
    }

    // Mark the given node as an output tensor.
    template<typename OutputView>
    void addOutput(Node node, const OutputView& output) {
      node->set_output(true)
      std::vector<int64_t> strides;
      for(int i = 0; i < OutputView::rank; i++) {
        strides.push_back(output.stride(0));
      }
      node->set_stride(strides);
      outputTensors.push_back(node)
      boundOutputData.push_back(nullptr);
    }

    Node matmul(Node A, Node B) {
      return graph.matmul(A, B, fe::graph::Matmul_attributes());
    }

    Node conv2D(Node A, Node W, int padX, int padY, int strideX, int strideY, int dilationX, int dilationY) {
      auto conv_options =
        fe::graph::Conv_fprop_attributes().set_padding({padX, padY}).set_stride({strideX, strideY}).set_dilation({dilationX, dilationY});
      return graph->conv_fprop(A, W, conv_options);
    }

    Node bias(Node A, Node B) {
      //auto B = graph->tensor(fe::graph::Tensor_attributes().set_name("bias").set_dim({1, k, 1, 1}).set_stride({k, 1, 1, 1}));
      auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
      return graph->pointwise(A, B, bias_options);
    }

    Node relu(Node A) {
      auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
      return graph->pointwise(A, B, bias_options);
    }

    // After adding all nodes and marking the output tensors, finish building the graph.
    void build() {
      KKDNN_ASSERT(graph->build_operation_graph(handle).is_good());
      KKDNN_ASSERT(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
      KKDNN_ASSERT(graph->check_support().is_good());
      KKDNN_ASSERT(graph->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());
      // Determine workspace size and allocate it
      int64_t workspaceSize = 0;
      KKDNN_ASSERT(graph->get_workspace_size(workspaceSize).is_good());
      workspace = Workspace(Kokkos::view_alloc(Kokkos::WithoutInitializing("graph workspace")), workspaceSize);
      this->graphIsBuilt = true;
    }

    // Bind an input view to the graph at the given index
    template<typename InputView>
    void bindInput(int index, const InputView& input) {
      if(index >= (int) inputTensors.size())
        throw std::invalid_argument("Attempted to bind input tensor at out of bounds index");
      boundInputData[index] = input.data();
    } 

    // Bind an output view to the graph at the given index
    template<typename OutputView>
    void bindOutput(int index, const OutputView& output) {
      if(index >= (int) outputTensors.size())
        throw std::invalid_argument("Attempted to bind output tensor at out of bounds index");
      boundOutputData[index] = output.data();
    }

    void execute() {
      if(!this->graphIsBuilt) {
        throw std::runtime_error("Cannot execute graph before calling build()");
      }
      // Populate all pointers for input, output and constant tensors for the graph
      std::unordered_map<int64_t, void*> tensorPointers;
      for(size_t i = 0; i < inputTensors.size(); i++) {
        tensorPointers[inputTensors[i]->get_uid()] = boundInputData[i];
      }
      for(size_t i = 0; i < outputTensors.size(); i++) {
        tensorPointers[outputTensors[i]->get_uid()] = boundOutputData[i];
      }
      for(auto& constant : constantTensors) {
        tensorPointers[constant.first->get_uid()] = constant.second;
      }
      KKDNN_ASSERT(graph->execute(*handle, tensorPointers, workspace.data()).is_good());
    }
  };
}

#endif

#endif // KOKKOSDNN_HPP

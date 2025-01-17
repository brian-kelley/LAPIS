"#include <Kokkos_Core.hpp>\n"
"#include <type_traits>\n"
"#include <cstdint>\n"
"#include <unistd.h>\n"
"#include <iostream>\n"
"\n"
"template <typename T, int N>\n"
"struct StridedMemRefType {\n"
"  T *basePtr;\n"
"  T *data;\n"
"  int64_t offset;\n"
"  int64_t sizes[N];\n"
"  int64_t strides[N];\n"
"};\n"
"\n"
"namespace LAPIS\n"
"{\n"
"  using TeamPolicy = Kokkos::TeamPolicy<>;\n"
"  using TeamMember = typename TeamPolicy::member_type;\n"
"\n"
"  template<typename V>\n"
"    StridedMemRefType<typename V::value_type, V::rank> viewToStridedMemref(const V& v)\n"
"    {\n"
"      StridedMemRefType<typename V::value_type, V::rank> smr;\n"
"      smr.basePtr = v.data();\n"
"      smr.data = v.data();\n"
"      smr.offset = 0;\n"
"      for(int i = 0; i < int(V::rank); i++)\n"
"      {\n"
"        smr.sizes[i] = v.extent(i);\n"
"        smr.strides[i] = v.stride(i);\n"
"      }\n"
"      return smr;\n"
"    }\n"
"\n"
"  template<typename V>\n"
"    V stridedMemrefToView(const StridedMemRefType<typename V::value_type, V::rank>& smr)\n"
"    {\n"
"      using Layout = typename V::array_layout;\n"
"      static_assert(std::is_same_v<typename V::memory_space, Kokkos::HostSpace> ||\n"
"          std::is_same_v<typename V::memory_space, Kokkos::AnonymousSpace>,\n"
"          \"Can only convert a StridedMemRefType to a Kokkos::View in HostSpace.\");\n"
"      if constexpr(std::is_same_v<Layout, Kokkos::LayoutStride>)\n"
"      {\n"
"        size_t extents[8] = {0};\n"
"        size_t strides[8] = {0};\n"
"        for(int i = 0; i < V::rank; i++) {\n"
"          extents[i] = smr.sizes[i];\n"
"          strides[i] = smr.strides[i];\n"
"        }\n"
"        Layout layout(\n"
"            extents[0], strides[0],\n"
"            extents[1], strides[1],\n"
"            extents[2], strides[2],\n"
"            extents[3], strides[3],\n"
"            extents[4], strides[4],\n"
"            extents[5], strides[5],\n"
"            extents[6], strides[6],\n"
"            extents[7], strides[7]);\n"
"        return V(&smr.data[smr.offset], layout);\n"
"      }\n"
"      size_t extents[8] = {0};\n"
"      for(int i = 0; i < V::rank; i++)\n"
"        extents[i] = smr.sizes[i];\n"
"      Layout layout(\n"
"          extents[0], extents[1], extents[2], extents[3],\n"
"          extents[4], extents[5], extents[6], extents[7]);\n"
"      if constexpr(std::is_same_v<Layout, Kokkos::LayoutLeft>)\n"
"      {\n"
"        int64_t expectedStride = 1;\n"
"        for(int i = 0; i < int(V::rank); i++)\n"
"        {\n"
"          if(expectedStride != smr.strides[i])\n"
"            Kokkos::abort(\"Cannot convert non-contiguous StridedMemRefType to LayoutLeft Kokkos::View\");\n"
"          expectedStride *= smr.sizes[i];\n"
"        }\n"
"      }\n"
"      else if constexpr(std::is_same_v<Layout, Kokkos::LayoutRight>)\n"
"      {\n"
"        int64_t expectedStride = 1;\n"
"        for(int i = int(V::rank) - 1; i >= 0; i--)\n"
"        {\n"
"          if(expectedStride != smr.strides[i])\n"
"            Kokkos::abort(\"Cannot convert non-contiguous StridedMemRefType to LayoutRight Kokkos::View\");\n"
"          expectedStride *= smr.sizes[i];\n"
"        }\n"
"      }\n"
"      return V(&smr.data[smr.offset], layout);\n"
"    }\n"
"\n"
"  struct DualViewBase\n"
"  {\n"
"    virtual ~DualViewBase() {}\n"
"    virtual void syncHost() = 0;\n"
"    virtual void syncDevice() = 0;\n"
"    bool modified_host = false;\n"
"    bool modified_device = false;\n"
"    DualViewBase* parent;\n"
"  };\n"
"\n"
"  template<typename DataType, typename Layout>\n"
"    struct DualView : public DualViewBase\n"
"  {\n"
"    using HostView = Kokkos::View<DataType, Layout, Kokkos::DefaultHostExecutionSpace>;\n"
"    using DeviceView = Kokkos::View<DataType, Layout, Kokkos::DefaultExecutionSpace>;\n"
"\n"
"    static constexpr bool deviceAccessesHost = Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace, typename DeviceView::memory_space>::accessible;\n"
"    static constexpr bool hostAccessesDevice = Kokkos::SpaceAccessibility<Kokkos::DefaultHostExecutionSpace, typename DeviceView::memory_space>::accessible;\n"
"\n"
"    // Default constructor makes empty views and self as parent.\n"
"    DualView() : device_view(), host_view() {\n"
"      parent = this;\n"
"    }\n"
"\n"
"    // Constructor for allocating a new view.\n"
"    // Does not actually allocate anything yet; instead \n"
"    DualView(\n"
"        const std::string& label,\n"
"        size_t ex0 = KOKKOS_INVALID_INDEX, size_t ex1 = KOKKOS_INVALID_INDEX, size_t ex2 = KOKKOS_INVALID_INDEX, size_t ex3 = KOKKOS_INVALID_INDEX,\n"
"        size_t ex4 = KOKKOS_INVALID_INDEX, size_t ex5 = KOKKOS_INVALID_INDEX, size_t ex6 = KOKKOS_INVALID_INDEX, size_t ex7 = KOKKOS_INVALID_INDEX)\n"
"    {\n"
"      if constexpr(hostAccessesDevice) {\n"
"        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + \"_dev\"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);\n"
"        host_view = HostView(device_view.data(), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);\n"
"      }\n"
"      else if constexpr(deviceAccessesHost) {\n"
"        // Otherwise, host_view must be a separate allocation.\n"
"        host_view = HostView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + \"_host\"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);\n"
"        device_view = DeviceView(host_view.data(), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);\n"
"      }\n"
"      else {\n"
"        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + \"_dev\"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);\n"
"        host_view = HostView(Kokkos::view_alloc(Kokkos::WithoutInitializing, label + \"_host\"), ex0, ex1, ex2, ex3, ex4, ex5, ex6, ex7);\n"
"      }\n"
"      parent = this;\n"
"    }\n"
"\n"
"    // Constructor which is given explicit device and host views, and a parent.\n"
"    // This can be used for subviewing/casting operations.\n"
"    // Note: d,h should alias parent\'s memory, but they can\n"
"    // have a different data type and layout.\n"
"    DualView(DeviceView d, HostView h, DualViewBase* parent_)\n"
"      : device_view(d), host_view(h)\n"
"    {\n"
"      parent = parent_;\n"
"      // Walk up to the top-level DualView (which has itself as parent).\n"
"      // This is important because its modify flags must be used for itself and all children.\n"
"      // Children have their own flag members, but they are not used or kept in sync with parent.\n"
"      while(parent->parent != parent)\n"
"        parent = parent->parent;\n"
"    }\n"
"\n"
"    // Constructor for a device view from an external source (e.g. Kokkos-based application)\n"
"    DualView(DeviceView d)\n"
"    {\n"
"      modified_device = true;\n"
"      if constexpr(deviceAccessesHost) {\n"
"        host_view = HostView(d.data(), d.layout());\n"
"      }\n"
"      else {\n"
"        host_view = HostView(Kokkos::view_alloc(Kokkos::WithoutInitializing, d.label() + \"_host\"), d.layout());\n"
"      }\n"
"      device_view = d;\n"
"      parent = this;\n"
"    }\n"
"\n"
"    // Constructor for a host view from an external source (e.g. python).\n"
"    // Use SFINAE to enable this only when DeviceView and HostView have different types/spaces,\n"
"    // since otherwise it would be a duplicate definition of the DeviceView constructor above.\n"
"    DualView(HostView h, typename std::enable_if_t<!std::is_same_v<DeviceView, HostView>>* = nullptr)\n"
"    {\n"
"      modified_host = true;\n"
"      if constexpr(deviceAccessesHost) {\n"
"        device_view = DeviceView(h.data(), h.layout());\n"
"      }\n"
"      else {\n"
"        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, h.label() + \"_dev\"), h.layout());\n"
"      }\n"
"      host_view = h;\n"
"      parent = this;\n"
"    }\n"
"\n"
"    // Copy-assignment equivalent to the above constructor.\n"
"    // Shallow copying a temporary DualView to a persistent one leaves the\n"
"    // persistent one in an invalid state, since its parent pointer still points to the temporary.\n"
"    //\n"
"    // Shallow-copy from one persistent DualView to another persistent or temporary is OK, as long\n"
"    // as the lifetime of original covers the lifetime of the copy.\n"
"    DualView& operator=(const HostView& h)\n"
"    {\n"
"      modified_host = true;\n"
"      if constexpr(deviceAccessesHost) {\n"
"        device_view = DeviceView(h.data(), h.layout());\n"
"      }\n"
"      else {\n"
"        device_view = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, h.label() + \"_dev\"), h.layout());\n"
"      }\n"
"      host_view = h;\n"
"      parent = this;\n"
"      return *this;\n"
"    }\n"
"\n"
"    // General shallow-copy from one DualView to another\n"
"    // (used by static -> dynamic conversion)\n"
"    template<typename OtherData, typename OtherLayout>\n"
"    DualView(const DualView<OtherData, OtherLayout>& other)\n"
"    {\n"
"      device_view = other.device_view;\n"
"      host_view = other.host_view;\n"
"      parent = other.parent;\n"
"    }\n"
"\n"
"    void modifyHost()\n"
"    {\n"
"      parent->modified_host = true;\n"
"    }\n"
"\n"
"    void modifyDevice()\n"
"    {\n"
"      parent->modified_device = true;\n"
"    }\n"
"\n"
"    bool modifiedHost()\n"
"    {\n"
"      // note: parent may just point to this\n"
"      return parent->modified_host;\n"
"    }\n"
"\n"
"    bool modifiedDevice()\n"
"    {\n"
"      // note: parent may just point to this\n"
"      return parent->modified_device;\n"
"    }\n"
"\n"
"    void syncHost() override\n"
"    {\n"
"      if (device_view.data() == host_view.data()) {\n"
"        // Imitating Kokkos::DualView behavior: if device and host are the same space\n"
"        // then this sync (if required) is equivalent to a fence.\n"
"        if(parent->modified_device) {\n"
"          parent->modified_device = false;\n"
"          Kokkos::fence();\n"
"        }\n"
"      }\n"
"      else if (parent->modified_device) {\n"
"        if(parent == this) {\n"
"          Kokkos::deep_copy(host_view, device_view);\n"
"          modified_device = false;\n"
"        }\n"
"        else {\n"
"          parent->syncHost();\n"
"        }\n"
"      }\n"
"    }\n"
"\n"
"    void syncDevice() override\n"
"    {\n"
"      // If host and device views are the same, do not sync or fence\n"
"      // because all host execution spaces are synchronous.\n"
"      // Any changes on the host side are immediately visible on the device side.\n"
"      if (device_view.data() != host_view.data()) {\n"
"        if(parent == this) {\n"
"          if(modified_host) {\n"
"            Kokkos::deep_copy(device_view, host_view);\n"
"            modified_host = false;\n"
"          }\n"
"        }\n"
"        else {\n"
"          parent->syncDevice();\n"
"        }\n"
"      }\n"
"    }\n"
"\n"
"    void deallocate() {\n"
"      device_view = DeviceView();\n"
"      host_view = HostView();\n"
"    }\n"
"\n"
"    DeviceView device_view;\n"
"    HostView host_view;\n"
"  };\n"
"\n"
"  inline int threadParallelVectorLength(int par) {\n"
"    if (par < 1)\n"
"      return 1;\n"
"    int max_vector_length = TeamPolicy::vector_length_max();\n"
"    int vector_length = 1;\n"
"    while(vector_length < max_vector_length && vector_length * 6 < par) vector_length *= 2;\n"
"    return vector_length;\n"
"  }\n"
"\n"
"} // namespace LAPIS\n"
"\n"
""

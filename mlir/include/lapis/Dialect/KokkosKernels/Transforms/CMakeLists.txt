set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name KokkosKernels)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix KokkosKernels)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix KokkosKernels)
add_public_tablegen_target(MLIRKokkosKernelsPassIncGen)

add_mlir_doc(Passes KokkosKernelsPasses ./ -gen-pass-doc)

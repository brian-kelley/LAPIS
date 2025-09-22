//===-- Passes.td - KokkosKernels pass definition file -----*- tablegen -*-===//

#ifndef MLIR_DIALECT_KOKKOSKERNELS_TRANSFORMS_PASSES
#define MLIR_DIALECT_KOKKOSKERNELS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToKokkosKernels : Pass<"linalg-to-kokkoskernels", "ModuleOp"> {
  let summary = "Rewrites linalg ops using the KokkosKernels dialect where possible";
  let description = [{
  }];
  let constructor = "mlir::createLinalgToKokkosKernelsPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "sparse_tensor::TensorDialect",
    "memref::MemRefDialect",
    "func::FuncDialect"
  ];
}

#endif // MLIR_DIALECT_KOKKOSKERNELS_TRANSFORMS_PASSES

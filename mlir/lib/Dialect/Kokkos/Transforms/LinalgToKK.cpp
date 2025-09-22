//===- LinalgToKK.cpp -===//

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::kokkos;

namespace mlir {
#define GEN_PASS_DEF_LINALGTOKKPASS
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"
}

// Utility functions from mlir/lib/Dialect/SparseTensor/Transforms/SparseGPUCodegen.cpp
// to allow checking tensor storage types

static bool isCsrTensor(Value v) {
  auto aTp = sparse_tensor::getSparseTensorType(v);
  return aTp.getDimRank() == 2 && aTp.getLvlRank() == 2 && aTp.isIdentity() &&
         aTp.isDenseLvl(0) && aTp.isCompressedLvl(1) && aTp.isOrderedLvl(1) &&
         aTp.isUniqueLvl(1);
}

static bool isDenseTensor(Value v) {
  auto sTp = sparse_tensor::getSparseTensorType(v);
  return sTp.getDimRank() == sTp.getLvlRank() && sTp.isAllDense();
}

struct LinalgToKKPass
    : public impl::LinalgToKKPassBase<LinalgToKKPass> {

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    func::FuncOp func = getOperation();
    // Scan through for each supported op type
    func.walk<WalkOrder::PostOrder>([&](linalg::MatmulOp matmul) {
        llvm::outs() << "Found matmul op\n";
        auto loc = matmul.getLoc();
        rewriter.setInsertionPoint(matmul);
        Value A = matmul.getInputs()[0];
        Value B = matmul.getInputs()[1];
        Value C = matmul.getOutputs()[0];
        if(isDenseTensor(A) && isDenseTensor(B) && isDenseTensor(C)) {
          llvm::outs() << "Operands dense so rewriting as gemm\n";
          //Rewrite as GEMM call
          SmallVector<Value> args;
          args.push_back(A);
          args.push_back(B);
          args.push_back(C);
          auto gemm = rewriter.create<kokkos::GemmOp>(loc, C.getType(), A, B, C);
          rewriter.replaceOp(matmul, gemm);
          //auto result = rewriter.create<emitc::CallOpaqueOp>(loc, C.getType(), "LAPIS::gemm", args).getResult(0);
          //rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(matmul, result);
        }
        else if(isCsrTensor(A) && isDenseTensor(B) && isDenseTensor(C)) {
          llvm::outs() << "Operand 0 sparse so rewriting as spmv\n";
          //Rewrite as SpMM (spmv rank-2) call
          auto spmv = rewriter.create<kokkos::SpmvOp>(loc, C.getType(), A, B, C);
          rewriter.replaceOp(matmul, spmv);
        }
    });
    func.walk<WalkOrder::PostOrder>([&](linalg::MatvecOp matvec) {
        llvm::outs() << "Found matvec op\n";
        auto loc = matvec.getLoc();
        rewriter.setInsertionPoint(matvec);
        Value A = matvec.getInputs()[0];
        Value x = matvec.getInputs()[1];
        Value y = matvec.getOutputs()[0];
        if(isDenseTensor(A) && isDenseTensor(x) && isDenseTensor(y)) {
          llvm::outs() << "Rewriting to gemv\n";
          //Rewrite as GEMV call
          auto call = rewriter.create<kokkos::GemvOp>(loc, y.getType(), A, x, y);
          rewriter.replaceOp(matvec, call);
        }
        else if(isCsrTensor(A) && isDenseTensor(x) && isDenseTensor(y)) {
          llvm::outs() << "Rewriting to spmv\n";
          //Rewrite as SpMV call
          auto spmv = rewriter.create<kokkos::SpmvOp>(loc, y.getType(), A, x, y);
          rewriter.replaceOp(matvec, spmv);
        }
    });
  }
};

std::unique_ptr<Pass> mlir::createLinalgToKKPass() {
  return std::make_unique<LinalgToKKPass>();
}


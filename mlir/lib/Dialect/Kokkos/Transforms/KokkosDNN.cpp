//===- LinalgToKK.cpp -===//

#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::kokkos;

namespace mlir {
#define GEN_PASS_DEF_KOKKOSDNN
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

/// Helper to detect a * b with arguments taken from given block.
static bool matchMulOfArgs(Block *block, Value val) {
  if (auto *def = val.getDefiningOp()) {
    if (isa<arith::MulFOp, arith::MulIOp>(def)) {
      Value a = block->getArguments()[0];
      Value b = block->getArguments()[1];
      return (def->getOperand(0) == a && def->getOperand(1) == b) ||
             (def->getOperand(0) == b && def->getOperand(1) == a);
    }
  }
  return false;
}

static bool matchReLU(linalg::GenericOp op) {
  auto yieldOp = cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (auto *def = yieldOp.getOperand(0).getDefiningOp()) {
    if (isa<arith::SelectOp>(def)) {
      return true;
    }
  }
  return false;
}

static bool matchBias(linalg::GenericOp op) {
  auto yieldOp = cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (auto *def = yieldOp.getOperand(0).getDefiningOp()) {
    if (isa<arith::AddFOp>(def)) {
      Value a = op.getBlock()->getArguments()[0];
      Value b = op.getBlock()->getArguments()[1];
      return (def->getOperand(0) == a && def->getOperand(1) == b) ||
             (def->getOperand(0) == b && def->getOperand(1) == a);
    }
  }
  return false;
}

static bool matchBatchNorm(linalg::GenericOp op) {
  // Batch norm contains an rsqrt in the body
  bool containsRsqrt = false;
  op.walk<WalkOrder::PostOrder>([&](math::RsqrtOp op) {
      containsRsqrt  = true;
  });
  return containsRsqrt;
}

// Match the first GenericOp emitted for adaptive average pool
static bool matchAvgPool(linalg::GenericOp op) {
  int numSelects = 0;
  op.walk<WalkOrder::PostOrder>([&](arith::SelectOp op) {
      numSelects++;
  });
  return numSelects == 2;
}

/*
static bool matchSumOfMultOfArgs(linalg::GenericOp op) {
  auto yieldOp = cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (auto *def = yieldOp.getOperand(0).getDefiningOp()) {
    if (isa<arith::AddFOp, arith::AddIOp>(def)) {
      Value x = op.getBlock()->getArguments()[2];
      return (def->getOperand(0) == x &&
              matchMulOfArgs(op.getBlock(), def->getOperand(1))) ||
             (def->getOperand(1) == x &&
              matchMulOfArgs(op.getBlock(), def->getOperand(0)));
    }
  }
  return false;
}
*/

struct KokkosDNNPass
    : public impl::KokkosDNNBase<KokkosDNNPass> {

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    func::FuncOp func = getOperation();

    // Scan through for each supported op type
    func.walk<WalkOrder::PostOrder>([&](linalg::GenericOp op) {
      auto loc = op.getLoc();
      rewriter.setInsertionPoint(op);

      /*
      // Logic to detect spmv, spmm, gemm, gemv taken from SparseGPUCodegen.cpp
      if (op.getNumDpsInits() != 1)
        return; // reject multi-output

      const unsigned numLoops = op.getNumLoops();
      const unsigned numTensors = op->getNumOperands();
      const auto iteratorTypes = op.getIteratorTypesArray();
      SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();

      using MapList = ArrayRef<ArrayRef<AffineExpr>>;
      auto infer = [&](MapList m) {
        return AffineMap::inferFromExprList(m, op.getContext());
      };
      AffineExpr i, j, k;
      bindDims(&getContext(), i, j, k);

      // Recognize a SpMV kernel.
      if (numLoops == 2 && numTensors == 3 &&
          linalg::isParallelIterator(iteratorTypes[0]) &&
          linalg::isReductionIterator(iteratorTypes[1]) &&
          maps == infer({{i, j}, {j}, {i}}) && matchSumOfMultOfArgs(op)) {
        auto A = op.getOperand(0);
        auto x = op.getOperand(1);
        auto yin = op.getOperand(2);
        if (runOnSparse && isCsrTensor(A) && isDenseTensor(x) && isDenseTensor(yin)) {
          // spmv 
          auto spmv = rewriter.create<kokkos::SpmvTensorOp>(loc, yin.getType(), A, x, yin);
          rewriter.replaceOp(op, spmv);
        }
        else if (runOnDense && isDenseTensor(A) && isDenseTensor(x) && isDenseTensor(yin)) {
          // gemv
          auto gemv = rewriter.create<kokkos::GemvOp>(loc, yin.getType(), A, x, yin);
          rewriter.replaceOp(op, gemv);
        }
      }
      // Recognize a GEMM or SpMM kernel.
      else if (numLoops == 3 && numTensors == 3 &&
          linalg::isParallelIterator(iteratorTypes[0]) &&
          linalg::isParallelIterator(iteratorTypes[1]) &&
          linalg::isReductionIterator(iteratorTypes[2]) &&
          maps == infer({{i, k}, {k, j}, {i, j}}) && matchSumOfMultOfArgs(op)) {
        auto A = op.getOperand(0);
        auto B = op.getOperand(1);
        auto Cin = op.getOperand(2);
        if (runOnSparse && isCsrTensor(A) && isDenseTensor(B) && isDenseTensor(Cin)) {
          // SpMM
          auto spmm = rewriter.create<kokkos::SpmvTensorOp>(loc, Cin.getType(), A, B, Cin);
          rewriter.replaceOp(op, spmm);
        }
        else if (runOnDense && isDenseTensor(A) && isDenseTensor(B) && isDenseTensor(Cin)) {
          // GEMM
          auto gemm = rewriter.create<kokkos::GemmOp>(loc, Cin.getType(), A, B, Cin);
          rewriter.replaceOp(op, gemm);
        }
      }
      */
    });
  }
};

std::unique_ptr<Pass> mlir::createKokkosDNNPass() {
  return std::make_unique<KokkosDNNPass>();
}


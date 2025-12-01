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

static Value getBatchNormEps(linalg::GenericOp op) {
  Value v;
  op.walk<WalkOrder::PostOrder>([&](arith::TruncFOp trunc) {
    v = trunc.getOperand();
  });
  return v;
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
      // Check for possible kernels
      if(matchBias(op)) {
        Type resultType = op->getResult(0).getType();
        Value input = op->getOperand(0);
        Value bias = op->getOperand(1);
        auto newOp = rewriter.create<kokkos::BiasOp>(loc, resultType, input, bias);
        rewriter.replaceOp(op, newOp);
      }
      else if(matchReLU(op)) {
        Type resultType = op->getResult(0).getType();
        Value input = op->getOperand(0);
        auto newOp = rewriter.create<kokkos::ReLUOp>(loc, resultType, input);
        rewriter.replaceOp(op, newOp);
      }
      else if(matchBatchNorm(op)) {
        Type resultType = op->getResult(0).getType();
        Value input = op->getOperand(0);
        Value scale = op->getOperand(1);
        Value bias = op->getOperand(2);
        Value mean = op->getOperand(3);
        Value variance = op->getOperand(4);
        Value eps = getBatchNormEps(op);
        auto newOp = rewriter.create<kokkos::BatchNorm2DOp>(loc, resultType, input, scale, bias, mean, variance, eps);
        rewriter.replaceOp(op, newOp);
      }
    });
    // Scan for known op types
    func.walk<WalkOrder::PostOrder>([&](linalg::Conv2DNchwFchwOp op) {
      // %26 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%padded_106, %cst_27 : tensor<64x64x58x58xf32>, tensor<128x64x3x3xf32>) outs(%25 : tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xf32>
      auto loc = op.getLoc();
      rewriter.setInsertionPoint(op);
      Type resultType = op->getResult(0).getType();
      Value input = op->getOperand(0);
      Value weights = op->getOperand(1);
      // Check if input was the result of a pad.
      // If so, we want to fold the pad into the kokkos.conv2d op.
      // Otherwise, we assume the padding is 0 in both directions.
      tensor::PadOp padOp = dyn_cast<tensor::PadOp>(input.getDefiningOp());
      int padX = 0;
      int padY = 0;
      Value unpaddedInput = input;
      if(padOp) {
        unpaddedInput = padOp->getOperand(0);
        auto padLow = padOp.getStaticLow();
        auto padHigh = padOp.getStaticHigh();
        if(padLow.size() != 4U || padHigh.size() != 4U) {
          padOp.emitError("Expected tensor.pad (producing input to conv2d) to have 4D padding.");
          return;
        }
        // Make sure pad is only in the expected dimensions
        for(int i = 0; i < 2; i++) {
          if(padLow[i] != 0 || padHigh[i] != 0) {
            padOp.emitError("Did not expect tensor.pad to apply padding in batch/channel dimensions!");
            return;
          }
        }
        if(padLow[2] != padHigh[2] || padLow[3] != padHigh[3]) {
          padOp.emitError("Expect tensor.pad to apply same padding to each side of tensor!");
          return;
        }
        padX = padLow[2];
        padY = padLow[3];
      }
      // Get stride information
      SmallVector<int> strides;
      {
        auto stridesAttr = op.getStrides();
        for(auto it = stridesAttr.begin(); it != stridesAttr.end(); it++) {
          strides.push_back((int) (*it).getLimitedValue());
        }
      }
      llvm::outs() << "Extracted strides for conv2d: ";
      for(auto s : strides)
        llvm::outs() << s << ' ';
      llvm::outs() << '\n';
      // Create new op, bypassing tensor.pad if there was one
      auto newOp = rewriter.create<kokkos::Conv2DOp>(loc, resultType, unpaddedInput, weights, rewriter.getIndexAttr(strides[0]), rewriter.getIndexAttr(strides[1]), rewriter.getIndexAttr(padX), rewriter.getIndexAttr(padY));
      rewriter.replaceOp(op, newOp);
    });
    func.walk<WalkOrder::PostOrder>([&](linalg::MatmulOp op) {
      auto loc = op.getLoc();
      rewriter.setInsertionPoint(op);
      auto inputs = op.getInputs();
      auto newOp = rewriter.create<kokkos::MatmulOp>(loc, op.getResult(0).getType(), inputs[0], inputs[1]);
      rewriter.replaceOp(op, newOp);
    });
    // Delete all fill (zero-initialization) ops since kokkosDNN doesn't need it
    func.walk<WalkOrder::PostOrder>([&](linalg::FillOp op) {
      bool isZero = false;
      Value val = op.getInputs()[0];
      if(auto cst = dyn_cast<arith::ConstantOp>(val.getDefiningOp())) {
        Attribute attr = cst.getValue();
        if(auto fattr = dyn_cast<FloatAttr>(attr)) {
          isZero = fattr.getValue().isZero();
        }
      }
      if(isZero) {
        // Replace fill op's result usages by its input (an empty tensor)
        rewriter.replaceOp(op, op.getOutputs()[0]);
      }
    });

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
  }
};

std::unique_ptr<Pass> mlir::createKokkosDNNPass() {
  return std::make_unique<KokkosDNNPass>();
}


//===- ** Modified from original version in llvm-project ** -===//
//===- SparseTensorCodegen.cpp - Sparse tensor primitives conversion ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "lapis/Dialect/Kokkos/IR/KokkosDialect.h"
#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::kokkos;
using namespace mlir::sparse_tensor;

namespace mlir {
#define GEN_PASS_DEF_LAPISSPARSECODEGENPASS
#include "lapis/Dialect/Kokkos/Transforms/Passes.h.inc"
}

inline UnrealizedConversionCastOp getTuple(Value tensor) {
  return llvm::cast<UnrealizedConversionCastOp>(tensor.getDefiningOp());
}

static void flattenOperands(ValueRange operands,
                            SmallVectorImpl<Value> &flattened) {
  // In case of
  // sparse_tensor, c, sparse_tensor
  // ==>
  // memref ..., c, memref ...
  for (auto operand : operands) {
    if (getSparseTensorEncoding(operand.getType())) {
      auto tuple = getTuple(operand);
      // An unrealized_conversion_cast will be inserted by type converter to
      // inter-mix the gap between 1:N conversion between sparse tensors and
      // fields. In this case, take the operands in the cast and replace the
      // sparse tensor output with the flattened type array.
      flattened.append(tuple.getOperands().begin(), tuple.getOperands().end());
    } else {
      flattened.push_back(operand);
    }
  }
}

class SpmvConverter : public OpConversionPattern<kokkos::SpmvTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(kokkos::SpmvTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // In case of:
    //  sparse_tensor, f, sparse_tensor = call @foo(...)
    // ==>
    //  memref..., f, memref = call @foo(...) replace with
    //  cast(memref...)->sparse_tensor, f, cast(memref...)->sparse_tensor
    SmallVector<Type> finalRetTy;
    SmallVector<Type> resultTypes;
    resultTypes.push_back(op.getResult().getType());
    if (failed(typeConverter->convertTypes(resultTypes, finalRetTy)))
      return failure();

    // (1) Generates new call with flattened return value.
    SmallVector<Value> flattened;
    flattenOperands(adaptor.getOperands(), flattened);
    auto newSpmv = rewriter.create<kokkos::SpmvOp>(loc, finalRetTy, flattened);
    /*
    // (2) Create cast operation for sparse tensor returns.
    SmallVector<Value> castedRet;
    // Tracks the offset of current return value (of the original call)
    // relative to the new call (after sparse tensor flattening);
    unsigned retOffset = 0;
    // Temporal buffer to hold the flattened list of type for
    // a sparse tensor.
    SmallVector<Type> sparseFlat;
    for (auto ret : op.getResults()) {
      assert(retOffset < newSpmv.getNumResults());
      auto retType = ret.getType();
      if (failed(typeConverter->convertType(retType, sparseFlat)))
        llvm_unreachable("Failed to convert type in sparse tensor codegen");

      // Converted types can not be empty when the type conversion succeed.
      assert(!sparseFlat.empty());
      if (sparseFlat.size() > 1) {
        auto flatSize = sparseFlat.size();
        ValueRange fields(iterator_range<ResultRange::iterator>(
            newSpmv.result_begin() + retOffset,
            newSpmv.result_begin() + retOffset + flatSize));
        castedRet.push_back(genTuple(rewriter, loc, retType, fields));
        retOffset += flatSize;
      } else {
        // If this is an 1:1 conversion, no need for casting.
        castedRet.push_back(newSpmv.getResult(retOffset));
        retOffset++;
      }
      sparseFlat.clear();
    }
    */

    //assert(castedRet.size() == op.getNumResults());
    rewriter.replaceOp(op, newSpmv);
    return success();
  }
};

struct LAPISSparseCodegenPass
    : public impl::LAPISSparseCodegenPassBase<LAPISSparseCodegenPass> {

  bool createSparseDeallocs = false;

  LAPISSparseCodegenPass() = default;
  LAPISSparseCodegenPass(const LAPISSparseCodegenPass &pass) = default;
  LAPISSparseCodegenPass(bool createDeallocs, bool /* enableInit */) {
    createSparseDeallocs = createDeallocs;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    SparseTensorTypeToBufferConverter converter;
    ConversionTarget target(*ctx);
    // Most ops in the sparse dialect must go!
    target.addIllegalDialect<SparseTensorDialect>();
    target.addLegalOp<SortOp>();
    target.addLegalOp<PushBackOp>();
    // Storage specifier outlives sparse tensor pipeline.
    target.addLegalOp<GetStorageSpecifierOp>();
    target.addLegalOp<SetStorageSpecifierOp>();
    target.addLegalOp<StorageSpecifierInitOp>();
    // Note that tensor::FromElementsOp might be yield after lowering unpack.
    target.addLegalOp<tensor::FromElementsOp>();
    // All dynamic rules below accept new function, call, return, and
    // various tensor and bufferization operations as legal output of the
    // rewriting provided that all sparse tensor types have been fully
    // rewritten.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return converter.isSignatureLegal(op.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<bufferization::AllocTensorOp>(
        [&](bufferization::AllocTensorOp op) {
          return converter.isLegal(op.getType());
        });
    target.addDynamicallyLegalOp<bufferization::DeallocTensorOp>(
        [&](bufferization::DeallocTensorOp op) {
          return converter.isLegal(op.getTensor().getType());
        });
    // The following operations and dialects may be introduced by the
    // codegen rules, and are therefore marked as legal.
    target.addLegalOp<linalg::FillOp, linalg::YieldOp>();
    target.addLegalDialect<
        arith::ArithDialect, bufferization::BufferizationDialect,
        complex::ComplexDialect, memref::MemRefDialect, scf::SCFDialect, kokkos::KokkosDialect>();
    target.addIllegalOp<kokkos::SpmvTensorOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    // Populate with rules and apply rewriting rules.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    populateSparseTensorCodegenPatterns(converter, patterns, createSparseDeallocs, /* enableBufferInitialization */ false);
    // Add pattern to do the decomposition on kokkos.spmv_tensor ops to get kokkos.spmv ops
    //patterns.add<SpmvConverter>(converter, patterns.getContext());
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createLAPISSparseCodegenPass()
{
  return std::make_unique<LAPISSparseCodegenPass>(false, false);
}

std::unique_ptr<Pass> mlir::createLAPISSparseCodegenPass(bool createDeallocs)
{
  return std::make_unique<LAPISSparseCodegenPass>(createDeallocs, false);
}


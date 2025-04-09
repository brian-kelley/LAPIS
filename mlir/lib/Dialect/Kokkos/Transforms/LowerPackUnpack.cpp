//===- LowerPackUnpack.cpp - Patterns for lower-pack-unpack pass

#include "lapis/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace {
struct LowerPackPattern : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LowerPackPattern(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(tensor::PackOp op,
                                PatternRewriter &rewriter) const override {
    return result = linalg::lowerPack(rewriter, op);
  }
};

struct LowerUnpackPattern : public OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  LowerUnpackPattern(MLIRContext *context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(tensor::UnPackOp op,
                                PatternRewriter &rewriter) const override {
    return linalg::lowerUnPack(rewriter, op);
  }
};

} // namespace

void mlir::populateLowerPackUnpackPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerPackPattern>(patterns.getContext());
  patterns.add<LowerUnpackPattern>(patterns.getContext());
}


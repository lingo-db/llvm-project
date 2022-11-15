//===- ArithToEmitC.cpp - Arithmetic to EmitC dialect conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/ArithToEmitC/ConversionTarget.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOEMITCCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//
namespace {
struct ArithToEmitCConversionPass
    : public impl::ArithToEmitCConversionPassBase<ArithToEmitCConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    EmitCConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    mlir::arith::populateArithToEmitCConversionPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Straightforward Op Lowerings
//===----------------------------------------------------------------------===//

namespace {
  enum COperation {
    ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULO, FMOD, AND, OR, MAX, MIN, NEGATE,
    SHIFT_LEFT, SHIFT_RIGHT, XOR, TERNARY_OP, COMPARE_EQUALS, COMPARE_NOT_EQUALS,
    COMPARE_LESS, COMPARE_GREATER, COMPARE_LESS_EQ, COMPARE_GREATER_EQ,
    CEIL_DIV, FLOOR_DIV
  };

  std::unordered_map<COperation, std::string> opToFormatString{
      {COperation::ADD, "@0 + @1"},
      {COperation::SUBTRACT, "@0 - @1"},
      {COperation::MULTIPLY, "@0 * @1"},
      {COperation::DIVIDE, "@0 / @1"},
      {COperation::MODULO, "@0 % @1"},
      {COperation::FMOD, "fmod(@0, @1)"},
      {COperation::AND, "@0 & @1"},
      {COperation::OR, "@0 | @1"},
      {COperation::MAX, "@0 > @1 ? @0 : @1"},
      {COperation::MIN, "@0 < @1 ? @0 : @1"},
      {COperation::NEGATE, "-@0"},
      {COperation::SHIFT_LEFT, "@0 << @1"},
      {COperation::SHIFT_RIGHT, "@0 >> @1"},
      {COperation::XOR, "@0 ^ @1"},
      {COperation::TERNARY_OP, "@0 ? @1 : @2"},
      {COperation::COMPARE_EQUALS, "@0 == @1"},
      {COperation::COMPARE_NOT_EQUALS, "@0 != @1"},
      {COperation::COMPARE_LESS, "@0 < @1"},
      {COperation::COMPARE_LESS_EQ, "@0 <= @1"},
      {COperation::COMPARE_GREATER, "@0 > @1"},
      {COperation::COMPARE_GREATER_EQ, "@0 >= @1"},
      {COperation::CEIL_DIV, "@0 / @1 + ((@0 % @1)>0)"},
      {COperation::FLOOR_DIV, "@0 / @1 - ((@0 % @1 < 0)"},
  };

  template <typename ArithmeticOp, COperation cOp>
  LogicalResult GenericOpLowering(ArithmeticOp op, PatternRewriter &rewriter){
    rewriter.replaceOpWithNewOp<emitc::GenericOp>(
        op, op->getResult(0).getType(), opToFormatString[cOp], op->getOperands());
    return success();
  }

  template <typename CastOp>
  LogicalResult CastOpLowering(CastOp op, PatternRewriter &rewriter){
    rewriter.replaceOpWithNewOp<emitc::CastOp>(
        op, op->getResult(0).getType(), op->getOperands());
    return success();
  }

  LogicalResult CompareFOpLowering(arith::CmpFOp op, PatternRewriter &rewriter){
    switch (op.getPredicate()){
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ:
      return GenericOpLowering<arith::CmpFOp, COperation::COMPARE_EQUALS>(op, rewriter);
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE:
      return GenericOpLowering<arith::CmpFOp, COperation::COMPARE_NOT_EQUALS>(op, rewriter);
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT:
      return GenericOpLowering<arith::CmpFOp, COperation::COMPARE_LESS>(op, rewriter);
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE:
      return GenericOpLowering<arith::CmpFOp, COperation::COMPARE_LESS_EQ>(op, rewriter);
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT:
      return GenericOpLowering<arith::CmpFOp, COperation::COMPARE_GREATER>(op, rewriter);
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE:
      return GenericOpLowering<arith::CmpFOp, COperation::COMPARE_GREATER_EQ>(op, rewriter);
    default:
      failure();
    }
  }

  LogicalResult CompareIOpLowering(arith::CmpIOp op, PatternRewriter &rewriter){
    switch (op.getPredicate()){
    case arith::CmpIPredicate::eq:
      return GenericOpLowering<arith::CmpIOp, COperation::COMPARE_EQUALS>(op, rewriter);
    case arith::CmpIPredicate::ne:
      return GenericOpLowering<arith::CmpIOp, COperation::COMPARE_NOT_EQUALS>(op, rewriter);
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return GenericOpLowering<arith::CmpIOp, COperation::COMPARE_LESS>(op, rewriter);
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return GenericOpLowering<arith::CmpIOp, COperation::COMPARE_LESS_EQ>(op, rewriter);
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return GenericOpLowering<arith::CmpIOp, COperation::COMPARE_GREATER>(op, rewriter);
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return GenericOpLowering<arith::CmpIOp, COperation::COMPARE_GREATER_EQ>(op, rewriter);
    default:
      failure();
    }
  }
}



//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//
void mlir::arith::populateArithToEmitCConversionPatterns(mlir::RewritePatternSet &patterns) {
    patterns.add(GenericOpLowering<arith::AddFOp, COperation::ADD>);
    patterns.add(GenericOpLowering<arith::AddIOp, COperation::ADD>);
    // TODO: arith.addui_carry (::mlir::arith::AddUICarryOp)
    patterns.add(GenericOpLowering<arith::AndIOp, COperation::AND>);
    // TODO: arith.bitcast (::mlir::arith::BitcastOp)
    patterns.add(GenericOpLowering<arith::CeilDivSIOp, COperation::CEIL_DIV>);
    patterns.add(GenericOpLowering<CeilDivUIOp, COperation::CEIL_DIV>);
    patterns.add(CompareFOpLowering);
    patterns.add(CompareIOpLowering);
    // May remain: arith.constant (::mlir::arith::ConstantOp)
    patterns.add(GenericOpLowering<arith::DivFOp, COperation::DIVIDE>);
    // TODO: Attention treats leading bit as sign bit
    patterns.add(GenericOpLowering<arith::DivSIOp, COperation::DIVIDE>);
    patterns.add(GenericOpLowering<arith::DivUIOp, COperation::DIVIDE>);
    patterns.add(CastOpLowering<arith::ExtFOp>);
    patterns.add(CastOpLowering<arith::ExtSIOp>);
    patterns.add(CastOpLowering<arith::ExtUIOp>);
    patterns.add(CastOpLowering<arith::FPToSIOp>);
    patterns.add(CastOpLowering<arith::FPToUIOp>);
    patterns.add(GenericOpLowering<arith::FloorDivSIOp, COperation::FLOOR_DIV>);
    patterns.add(CastOpLowering<arith::IndexCastOp>);
    patterns.add(CastOpLowering<arith::IndexCastUIOp>);
    patterns.add(GenericOpLowering<arith::MaxFOp, COperation::MAX>);
    patterns.add(GenericOpLowering<arith::MaxSIOp, COperation::MAX>);
    patterns.add(GenericOpLowering<arith::MaxUIOp, COperation::MAX>);
    patterns.add(GenericOpLowering<arith::MinFOp, COperation::MIN>);
    patterns.add(GenericOpLowering<arith::MinSIOp, COperation::MIN>);
    patterns.add(GenericOpLowering<arith::MinUIOp, COperation::MIN>);
    patterns.add(GenericOpLowering<arith::MulFOp, COperation::MULTIPLY>);
    patterns.add(GenericOpLowering<arith::MulIOp, COperation::MULTIPLY>);
    patterns.add(GenericOpLowering<arith::NegFOp, COperation::NEGATE>);
    patterns.add(GenericOpLowering<arith::OrIOp, COperation::OR>);
    // TODO: Check for correctness and include header
    patterns.add(GenericOpLowering<arith::RemFOp, COperation::FMOD>);
    // TODO: arith.remsi (::mlir::arith::RemSIOp)
    patterns.add(GenericOpLowering<arith::RemUIOp, COperation::MODULO>);
    patterns.add(CastOpLowering<arith::SIToFPOp>);
    patterns.add(GenericOpLowering<arith::ShLIOp, COperation::SHIFT_LEFT>);
    // TODO check difference for negative numbers
    patterns.add(GenericOpLowering<arith::ShRSIOp, COperation::SHIFT_RIGHT>);
    patterns.add(GenericOpLowering<arith::ShRUIOp, COperation::SHIFT_RIGHT>);
    patterns.add(GenericOpLowering<arith::SubFOp, COperation::SUBTRACT>);
    patterns.add(GenericOpLowering<arith::SubIOp, COperation::SUBTRACT>);
    patterns.add(CastOpLowering<arith::TruncFOp>);
    // TODO: Probably wrong, top bits are discarded
    patterns.add(CastOpLowering<arith::TruncIOp>);
    patterns.add(CastOpLowering<arith::UIToFPOp>);
    patterns.add(GenericOpLowering<arith::XOrIOp, COperation::XOR>);
    patterns.add(GenericOpLowering<arith::SelectOp, COperation::TERNARY_OP>);
}

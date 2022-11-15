//===- ConversionTarget.cpp - Target for converting to the LLVM dialect ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToEmitC//ConversionTarget.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

mlir::EmitCConversionTarget::EmitCConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  this->addLegalDialect<emitc::EmitCDialect>();
  this->addIllegalDialect<arith::ArithDialect>();
  this->addLegalOp<arith::ConstantOp>();
}

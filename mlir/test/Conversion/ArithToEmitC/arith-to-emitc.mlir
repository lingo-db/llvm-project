// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-arith-to-emitc))" %s -split-input-file | FileCheck %s

// CHECK-LABEL: @testGeneric
func.func @testGeneric(f32, f32, i32, i32, ui32, ui32, i1) -> (f32, i32) {
^bb0(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32, %arg4: ui32, %arg5: ui32, %arg6 : i1):
// CHECK:  = emitc.generic "@0 - @1" %arg0, %arg1 : f32, f32 -> f32
  %0 = arith.subf %arg0, %arg1: f32
// CHECK: = emitc.generic "@0 - @1" %arg2, %arg3 : i32, i32 -> i32
  %1 = arith.subi %arg2, %arg3: i32
// CHECK: = emitc.generic "@0 & @1" %arg2, %arg3 : i32, i32 -> i32
  %2 = arith.andi %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 / @1 + ((@0 % @1)>0)" %arg2, %arg3 : i32, i32 -> i32
  %3 = arith.ceildivsi %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 / @1 + ((@0 % @1)>0)" %arg2, %arg3 : i32, i32 -> i32
  %4 = arith.ceildivui %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 / @1" %arg0, %arg1 : f32, f32 -> f32
  %5 = arith.divf %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 / @1" %arg2, %arg3 : i32, i32 -> i32
  %6 = arith.divsi %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 / @1" %arg2, %arg3 : i32, i32 -> i32
  %7 = arith.divui %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 / @1 - ((@0 % @1 < 0)" %arg2, %arg3 : i32, i32 -> i32
  %8 = arith.floordivsi %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 > @1 ? @0 : @1" %arg0, %arg1 : f32, f32 -> f32
  %9 = arith.maxf %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 > @1 ? @0 : @1" %arg2, %arg3 : i32, i32 -> i32
  %10 = arith.maxsi %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 > @1 ? @0 : @1" %arg2, %arg3 : i32, i32 -> i32
  %11 = arith.maxui %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 < @1 ? @0 : @1" %arg0, %arg1 : f32, f32 -> f32
  %12 = arith.minf %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 < @1 ? @0 : @1" %arg2, %arg3 : i32, i32 -> i32
  %13 = arith.minsi %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 < @1 ? @0 : @1" %arg2, %arg3 : i32, i32 -> i32
  %14 = arith.minui %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 * @1" %arg0, %arg1 : f32, f32 -> f32
  %15 = arith.mulf %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 * @1" %arg2, %arg3 : i32, i32 -> i32
  %16 = arith.muli %arg2, %arg3 : i32
// CHECK: = emitc.generic "-@0" %arg0 : f32 -> f32
  %17 = arith.negf %arg0 : f32
// CHECK: = emitc.generic "@0 | @1" %arg2, %arg3 : i32, i32 -> i32
  %18 = arith.ori %arg2, %arg3 : i32
// CHECK: = emitc.generic "fmod(@0, @1)" %arg0, %arg1 : f32, f32 -> f32
  %19 = arith.remf %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 % @1" %arg2, %arg3 : i32, i32 -> i32
  %20 = arith.remui %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 << @1" %arg2, %arg3 : i32, i32 -> i32
  %21 = arith.shli %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 >> @1" %arg2, %arg3 : i32, i32 -> i32
  %22 = arith.shrsi %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 >> @1" %arg2, %arg3 : i32, i32 -> i32
  %23 = arith.shrui %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 - @1" %arg0, %arg1 : f32, f32 -> f32
  %24 = arith.subf %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 - @1" %arg2, %arg3 : i32, i32 -> i32
  %25 = arith.subi %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 ^ @1" %arg2, %arg3 : i32, i32 -> i32
  %26 = arith.xori %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 ? @1 : @2" %arg6, %arg0, %arg1 : i1, f32, f32 -> f32
  %27 = arith.select %arg6, %arg0, %arg1 : f32

  func.return %arg0, %arg2: f32, i32
}


// CHECK-LABEL: @testCast
func.func @testCast(f32, f32, i32, i32, i64, f64, index) -> (f32, i32) {
^bb0(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32, %arg4: i64, %arg5: f64, %arg6: index):
// CHECK: = emitc.cast %arg0 : f32 to f64
  %0 = arith.extf %arg0 : f32 to f64
// CHECK: = emitc.cast %arg2 : i32 to i64
  %1 = arith.extsi %arg2 : i32 to i64
// CHECK: = emitc.cast %arg2 : i32 to i64
  %2 = arith.extui %arg2 : i32 to i64
// CHECK: = emitc.cast %arg0 : f32 to i32
  %3 = arith.fptosi %arg0 : f32 to i32
// CHECK: = emitc.cast %arg0 : f32 to i32
  %4 = arith.fptoui %arg0 : f32 to i32
// CHECK: = emitc.cast %arg6 : index to i32
  %5 = arith.index_cast %arg6 : index to i32
// CHECK: = emitc.cast %arg6 : index to i32
  %6 = arith.index_castui %arg6 : index to i32
// CHECK: = emitc.cast %arg2 : i32 to f32
  %7 = arith.sitofp %arg2 : i32 to f32
// CHECK: = emitc.cast %arg5 : f64 to f32
  %8 = arith.truncf %arg5 : f64 to f32
// CHECK: = emitc.cast %arg4 : i64 to i32
  %9 = arith.trunci %arg4 : i64 to i32
// CHECK: = emitc.cast %arg2 : i32 to f32
  %10 = arith.uitofp %arg2 : i32 to f32

  func.return %arg0, %arg2: f32, i32
}

// CHECK-LABEL: @testCompare
func.func @testCompare(f32, f32, i32, i32, ui32, ui32) -> (f32, i32) {
^bb0(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32, %arg4: ui32, %arg5: ui32):
// CHECK: = emitc.generic "@0 == @1" %arg0, %arg1 : f32, f32 -> i1
  %0 = arith.cmpf oeq, %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 < @1" %arg0, %arg1 : f32, f32 -> i1
  %1 = arith.cmpf olt, %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 <= @1" %arg0, %arg1 : f32, f32 -> i1
  %2 = arith.cmpf ole, %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 > @1" %arg0, %arg1 : f32, f32 -> i1
  %3 = arith.cmpf ogt, %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 >= @1" %arg0, %arg1 : f32, f32 -> i1
  %4 = arith.cmpf oge, %arg0, %arg1 : f32
// CHECK: = emitc.generic "@0 != @1" %arg0, %arg1 : f32, f32 -> i1
  %5 = arith.cmpf one, %arg0, %arg1 : f32

// CHECK: = emitc.generic "@0 == @1" %arg2, %arg3 : i32, i32 -> i1
  %6 = arith.cmpi eq, %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 < @1" %arg2, %arg3 : i32, i32 -> i1
  %7 = arith.cmpi slt, %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 <= @1" %arg2, %arg3 : i32, i32 -> i1
  %8 = arith.cmpi ule, %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 > @1" %arg2, %arg3 : i32, i32 -> i1
  %9 = arith.cmpi ugt, %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 >= @1" %arg2, %arg3 : i32, i32 -> i1
  %10 = arith.cmpi sge, %arg2, %arg3 : i32
// CHECK: = emitc.generic "@0 != @1" %arg2, %arg3 : i32, i32 -> i1
  %11 = arith.cmpi ne, %arg2, %arg3 : i32

  func.return %arg0, %arg2: f32, i32
}
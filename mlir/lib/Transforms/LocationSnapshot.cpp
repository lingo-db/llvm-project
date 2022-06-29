//===- LocationSnapshot.cpp - Location Snapshot Utilities -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/LocationSnapshot.h"
#include "PassDetail.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

#include<iostream>

using namespace mlir;

std::vector<std::string> split_string_by_newline(const std::string& str)
{
    auto result = std::vector<std::string>{};
    auto ss = std::stringstream{str};

    for (std::string line; std::getline(ss, line, '\n');)
        result.push_back(line);

    return result;
}
mlir::ArrayAttr getResultNames(mlir::MLIRContext* context,const std::string& line, size_t startCol,size_t count){
  llvm::StringRef resultArea=llvm::StringRef(line).substr(startCol,line.find("=",startCol)-startCol);
  auto colonPos=resultArea.find(":");
    std::vector<Attribute> attrs;
  if(colonPos!=llvm::StringRef::npos){
    llvm::StringRef baseName(resultArea.substr(0,colonPos));
    for(size_t i=0;i<count;i++){
      attrs.push_back(mlir::StringAttr::get(context,baseName.str()+"#"+std::to_string(i)));    
    }
  }else{
    size_t first=0;
    size_t last=0;
    size_t counter=0;
    while(first!=llvm::StringRef::npos){
      last=resultArea.find(",",first);
      auto effectiveLen=std::min(resultArea.size(),last)-1;
      std::string name=resultArea.substr(first,effectiveLen).str();
      attrs.push_back(mlir::StringAttr::get(context,name));    
      first=last==llvm::StringRef::npos?last:last+1;
    }
  }
  return ArrayAttr::get(context,attrs);
}

/// This function generates new locations from the given IR by snapshotting the
/// IR to the given stream, and using the printed locations within that stream.
/// If a 'tag' is non-empty, the generated locations are represented as a
/// NameLoc with the given tag as the name, and then fused with the existing
/// locations. Otherwise, the existing locations are replaced.
static void generateLocationsFromIR(raw_ostream &os, StringRef fileName,
                                    Operation *op, const OpPrintingFlags &flags,
                                    StringRef tag) {
  // Print the IR to the stream, and collect the raw line+column information.
  std::string outputStr;
  llvm::raw_string_ostream sstream(outputStr);
  AsmState::LocationMap opToLineCol;
  AsmState state(op, flags, &opToLineCol);
  op->print(sstream, state);
  os<<sstream.str();
  os.flush();
  auto lines=split_string_by_newline(outputStr);
  Builder builder(op->getContext());
  Optional<StringAttr> tagIdentifier;
  if (!tag.empty())
    tagIdentifier = builder.getStringAttr(tag);

  // Walk and generate new locations for each of the operations.
  StringAttr file = builder.getStringAttr(fileName);
  op->walk([&](Operation *opIt) {
    // Check to see if this operation has a mapped location. Some operations may
    // be elided from the printed form, e.g. the body terminators of some region
    // operations.
    auto it = opToLineCol.find(opIt);
    if (it == opToLineCol.end())
      return;
    const std::pair<unsigned, unsigned> &lineCol = it->second;
    mlir::Location newLoc = FileLineColLoc::get(file, lineCol.first, lineCol.second);
    if(opIt->getNumResults()){
      auto resNames=getResultNames(op->getContext(),lines[lineCol.first-1],lineCol.second,opIt->getNumResults());
      newLoc=NamedResultsLoc::get(resNames,newLoc);
    }
    // If we don't have a tag, set the location directly
    if (!tagIdentifier) {
      opIt->setLoc(newLoc);
      return;
    }

    // Otherwise, build a fused location with the existing op loc.
    opIt->setLoc(builder.getFusedLoc(
        {opIt->getLoc(), NameLoc::get(*tagIdentifier, newLoc)}));
  });
}

/// This function generates new locations from the given IR by snapshotting the
/// IR to the given file, and using the printed locations within that file. If
/// `filename` is empty, a temporary file is generated instead.
static LogicalResult generateLocationsFromIR(StringRef fileName, Operation *op,
                                             OpPrintingFlags flags,
                                             StringRef tag) {
  // If a filename wasn't provided, then generate one.
  SmallString<32> filepath(fileName);
  if (filepath.empty()) {
    if (std::error_code error = llvm::sys::fs::createTemporaryFile(
            "mlir_snapshot", "tmp.mlir", filepath)) {
      return op->emitError()
             << "failed to generate temporary file for location snapshot: "
             << error.message();
    }
  }

  // Open the output file for emission.
  std::string error;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      openOutputFile(filepath, &error);
  if (!outputFile)
    return op->emitError() << error;

  // Generate the intermediate locations.
  generateLocationsFromIR(outputFile->os(), filepath, op, flags, tag);
  outputFile->keep();
  return success();
}

/// This function generates new locations from the given IR by snapshotting the
/// IR to the given stream, and using the printed locations within that stream.
/// The generated locations replace the current operation locations.
void mlir::generateLocationsFromIR(raw_ostream &os, StringRef fileName,
                                   Operation *op, OpPrintingFlags flags) {
  ::generateLocationsFromIR(os, fileName, op, flags, /*tag=*/StringRef());
}
/// This function generates new locations from the given IR by snapshotting the
/// IR to the given file, and using the printed locations within that file. If
/// `filename` is empty, a temporary file is generated instead.
LogicalResult mlir::generateLocationsFromIR(StringRef fileName, Operation *op,
                                            OpPrintingFlags flags) {
  return ::generateLocationsFromIR(fileName, op, flags, /*tag=*/StringRef());
}

/// This function generates new locations from the given IR by snapshotting the
/// IR to the given stream, and using the printed locations within that stream.
/// The generated locations are represented as a NameLoc with the given tag as
/// the name, and then fused with the existing locations.
void mlir::generateLocationsFromIR(raw_ostream &os, StringRef fileName,
                                   StringRef tag, Operation *op,
                                   OpPrintingFlags flags) {
  ::generateLocationsFromIR(os, fileName, op, flags, tag);
}
/// This function generates new locations from the given IR by snapshotting the
/// IR to the given file, and using the printed locations within that file. If
/// `filename` is empty, a temporary file is generated instead.
LogicalResult mlir::generateLocationsFromIR(StringRef fileName, StringRef tag,
                                            Operation *op,
                                            OpPrintingFlags flags) {
  return ::generateLocationsFromIR(fileName, op, flags, tag);
}

namespace {
struct LocationSnapshotPass
    : public LocationSnapshotBase<LocationSnapshotPass> {
  LocationSnapshotPass() = default;
  LocationSnapshotPass(OpPrintingFlags flags, StringRef fileName, StringRef tag)
      : flags(flags) {
    this->fileName = fileName.str();
    this->tag = tag.str();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    if (failed(generateLocationsFromIR(fileName, op, flags, tag)))
      return signalPassFailure();
  }

  /// The printing flags to use when creating the snapshot.
  OpPrintingFlags flags;
};
} // namespace

std::unique_ptr<Pass> mlir::createLocationSnapshotPass(OpPrintingFlags flags,
                                                       StringRef fileName,
                                                       StringRef tag) {
  return std::make_unique<LocationSnapshotPass>(flags, fileName, tag);
}
std::unique_ptr<Pass> mlir::createLocationSnapshotPass() {
  return std::make_unique<LocationSnapshotPass>();
}

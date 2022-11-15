add_mlir_conversion_library(MLIRArithToEmitC
        ArithToEmitC.cpp
        ConversionTarget.cpp

        ADDITIONAL_HEADER_DIRS
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToEmitC

        DEPENDS
        MLIRConversionPassIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRArithDialect
        MLIREmitCDialect
        )
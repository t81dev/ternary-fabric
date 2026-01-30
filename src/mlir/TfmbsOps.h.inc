/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TfmbsOps.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::tfmbs {

/// Issue a DMA load into the fabric from host memory.
class TfmbsDmaLoadOp;

} // namespace mlir::tfmbs
namespace mlir::tfmbs {

/// High-level GEMV kernel targeting the ternary fabric.
class TfmbsGemvOp;

} // namespace mlir::tfmbs
namespace mlir::tfmbs {

/// Pack a dense tensor into PT-5 format.
/// Loads a host tensor and writes the quantized PT-5 representation into the
///     fabric's local memory.
class TfmbsPackOp;

} // namespace mlir::tfmbs
namespace mlir::tfmbs {

/// Direct copy between fabric memory banks.
class TfmbsTransferOp;

} // namespace mlir::tfmbs
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir::tfmbs {


//===----------------------------------------------------------------------===//
// ::mlir::tfmbs::TfmbsDmaLoadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class TfmbsDmaLoadOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  TfmbsDmaLoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tfmbs.dma_load", odsAttrs.getContext());
  }

  TfmbsDmaLoadOpGenericAdaptorBase(TfmbsDmaLoadOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class TfmbsDmaLoadOpGenericAdaptor : public detail::TfmbsDmaLoadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TfmbsDmaLoadOpGenericAdaptorBase;
  static constexpr int odsIndex_host = 0;
  static constexpr int odsIndex_fabric = 1;
public:
  TfmbsDmaLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TfmbsDmaLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TfmbsDmaLoadOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  TfmbsDmaLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : TfmbsDmaLoadOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  TfmbsDmaLoadOpGenericAdaptor(RangeT values, const TfmbsDmaLoadOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TfmbsDmaLoadOp, typename = std::enable_if_t<std::is_same_v<LateInst, TfmbsDmaLoadOp>>>
  TfmbsDmaLoadOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getHost() {
    return (*getODSOperands(0).begin());
  }

  ValueT getFabric() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TfmbsDmaLoadOpAdaptor : public TfmbsDmaLoadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TfmbsDmaLoadOpGenericAdaptor::TfmbsDmaLoadOpGenericAdaptor;
  TfmbsDmaLoadOpAdaptor(TfmbsDmaLoadOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TfmbsDmaLoadOp : public ::mlir::Op<TfmbsDmaLoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TfmbsDmaLoadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TfmbsDmaLoadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_host = 0;
  static constexpr int odsIndex_fabric = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfmbs.dma_load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getHost() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getFabric() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getHostMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getFabricMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value host, ::mlir::Value fabric);
  static TfmbsDmaLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value host, ::mlir::Value fabric);
  static TfmbsDmaLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value host, ::mlir::Value fabric);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value host, ::mlir::Value fabric);
  static TfmbsDmaLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value host, ::mlir::Value fabric);
  static TfmbsDmaLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value host, ::mlir::Value fabric);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TfmbsDmaLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TfmbsDmaLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TfmbsDmaLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TfmbsDmaLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::tfmbs
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfmbs::TfmbsDmaLoadOp)

namespace mlir::tfmbs {


//===----------------------------------------------------------------------===//
// ::mlir::tfmbs::TfmbsGemvOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class TfmbsGemvOpGenericAdaptorBase {
public:
  struct Properties {
    using telemetryTy = ::mlir::DictionaryAttr;
    telemetryTy telemetry;

    auto getTelemetry() const {
      auto &propStorage = this->telemetry;
      return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(propStorage);
    }
    void setTelemetry(const ::mlir::DictionaryAttr &propValue) {
      this->telemetry = propValue;
    }
    using tile_maskTy = ::mlir::IntegerAttr;
    tile_maskTy tile_mask;

    auto getTileMask() const {
      auto &propStorage = this->tile_mask;
      return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(propStorage);
    }
    void setTileMask(const ::mlir::IntegerAttr &propValue) {
      this->tile_mask = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.telemetry == this->telemetry &&
        rhs.tile_mask == this->tile_mask &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  TfmbsGemvOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tfmbs.gemv", odsAttrs.getContext());
  }

  TfmbsGemvOpGenericAdaptorBase(TfmbsGemvOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getTileMaskAttr();
  uint64_t getTileMask();
  ::mlir::DictionaryAttr getTelemetryAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(getProperties().telemetry);
    return attr;
  }

  ::std::optional< ::mlir::DictionaryAttr > getTelemetry();
};

} // namespace detail
template <typename RangeT>
class TfmbsGemvOpGenericAdaptor : public detail::TfmbsGemvOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TfmbsGemvOpGenericAdaptorBase;
  static constexpr int odsIndex_weight = 0;
  static constexpr int odsIndex_input = 1;
  static constexpr int odsIndex_output = 2;
public:
  TfmbsGemvOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TfmbsGemvOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TfmbsGemvOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  TfmbsGemvOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : TfmbsGemvOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  TfmbsGemvOpGenericAdaptor(RangeT values, const TfmbsGemvOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TfmbsGemvOp, typename = std::enable_if_t<std::is_same_v<LateInst, TfmbsGemvOp>>>
  TfmbsGemvOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getWeight() {
    return (*getODSOperands(0).begin());
  }

  ValueT getInput() {
    return (*getODSOperands(1).begin());
  }

  ValueT getOutput() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TfmbsGemvOpAdaptor : public TfmbsGemvOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TfmbsGemvOpGenericAdaptor::TfmbsGemvOpGenericAdaptor;
  TfmbsGemvOpAdaptor(TfmbsGemvOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TfmbsGemvOp : public ::mlir::Op<TfmbsGemvOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TfmbsGemvOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TfmbsGemvOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_weight = 0;
  static constexpr int odsIndex_input = 1;
  static constexpr int odsIndex_output = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("telemetry"), ::llvm::StringRef("tile_mask")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getTelemetryAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTelemetryAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getTileMaskAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getTileMaskAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfmbs.gemv");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getWeight() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getOutput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getWeightMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getOutputMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getTileMaskAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().tile_mask);
  }

  uint64_t getTileMask();
  ::mlir::DictionaryAttr getTelemetryAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::DictionaryAttr>(getProperties().telemetry);
  }

  ::std::optional< ::mlir::DictionaryAttr > getTelemetry();
  void setTileMaskAttr(::mlir::IntegerAttr attr) {
    getProperties().tile_mask = attr;
  }

  void setTileMask(::std::optional<uint64_t> attrValue);
  void setTelemetryAttr(::mlir::DictionaryAttr attr) {
    getProperties().telemetry = attr;
  }

  ::mlir::Attribute removeTileMaskAttr() {
      auto attr = getProperties().tile_mask;
      getProperties().tile_mask = {};
      return attr;
  }

  ::mlir::Attribute removeTelemetryAttr() {
      auto attr = getProperties().telemetry;
      getProperties().telemetry = {};
      return attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::DictionaryAttr telemetry);
  static TfmbsGemvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::DictionaryAttr telemetry);
  static TfmbsGemvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::DictionaryAttr telemetry);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::DictionaryAttr telemetry);
  static TfmbsGemvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::DictionaryAttr telemetry);
  static TfmbsGemvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::DictionaryAttr telemetry);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/uint64_t tile_mask, /*optional*/::mlir::DictionaryAttr telemetry);
  static TfmbsGemvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/uint64_t tile_mask, /*optional*/::mlir::DictionaryAttr telemetry);
  static TfmbsGemvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/uint64_t tile_mask, /*optional*/::mlir::DictionaryAttr telemetry);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/uint64_t tile_mask, /*optional*/::mlir::DictionaryAttr telemetry);
  static TfmbsGemvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/uint64_t tile_mask, /*optional*/::mlir::DictionaryAttr telemetry);
  static TfmbsGemvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/uint64_t tile_mask, /*optional*/::mlir::DictionaryAttr telemetry);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TfmbsGemvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TfmbsGemvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TfmbsGemvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TfmbsGemvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::tfmbs
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfmbs::TfmbsGemvOp)

namespace mlir::tfmbs {


//===----------------------------------------------------------------------===//
// ::mlir::tfmbs::TfmbsPackOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class TfmbsPackOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  TfmbsPackOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tfmbs.pack", odsAttrs.getContext());
  }

  TfmbsPackOpGenericAdaptorBase(TfmbsPackOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class TfmbsPackOpGenericAdaptor : public detail::TfmbsPackOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TfmbsPackOpGenericAdaptorBase;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_dst = 1;
public:
  TfmbsPackOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TfmbsPackOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TfmbsPackOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  TfmbsPackOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : TfmbsPackOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  TfmbsPackOpGenericAdaptor(RangeT values, const TfmbsPackOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TfmbsPackOp, typename = std::enable_if_t<std::is_same_v<LateInst, TfmbsPackOp>>>
  TfmbsPackOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  ValueT getDst() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TfmbsPackOpAdaptor : public TfmbsPackOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TfmbsPackOpGenericAdaptor::TfmbsPackOpGenericAdaptor;
  TfmbsPackOpAdaptor(TfmbsPackOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TfmbsPackOp : public ::mlir::Op<TfmbsPackOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TfmbsPackOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TfmbsPackOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_dst = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfmbs.pack");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getSrc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getDst() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getSrcMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDstMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dst);
  static TfmbsPackOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value src, ::mlir::Value dst);
  static TfmbsPackOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value src, ::mlir::Value dst);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst);
  static TfmbsPackOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst);
  static TfmbsPackOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TfmbsPackOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TfmbsPackOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TfmbsPackOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TfmbsPackOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::tfmbs
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfmbs::TfmbsPackOp)

namespace mlir::tfmbs {


//===----------------------------------------------------------------------===//
// ::mlir::tfmbs::TfmbsTransferOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class TfmbsTransferOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  TfmbsTransferOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tfmbs.transfer", odsAttrs.getContext());
  }

  TfmbsTransferOpGenericAdaptorBase(TfmbsTransferOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class TfmbsTransferOpGenericAdaptor : public detail::TfmbsTransferOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TfmbsTransferOpGenericAdaptorBase;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_dst = 1;
  static constexpr int odsIndex_len = 2;
public:
  TfmbsTransferOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TfmbsTransferOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TfmbsTransferOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  TfmbsTransferOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : TfmbsTransferOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  TfmbsTransferOpGenericAdaptor(RangeT values, const TfmbsTransferOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TfmbsTransferOp, typename = std::enable_if_t<std::is_same_v<LateInst, TfmbsTransferOp>>>
  TfmbsTransferOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  ValueT getDst() {
    return (*getODSOperands(1).begin());
  }

  ValueT getLen() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TfmbsTransferOpAdaptor : public TfmbsTransferOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TfmbsTransferOpGenericAdaptor::TfmbsTransferOpGenericAdaptor;
  TfmbsTransferOpAdaptor(TfmbsTransferOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TfmbsTransferOp : public ::mlir::Op<TfmbsTransferOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TfmbsTransferOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TfmbsTransferOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_dst = 1;
  static constexpr int odsIndex_len = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfmbs.transfer");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getSrc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getDst() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getLen() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getSrcMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDstMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getLenMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dst, ::mlir::Value len);
  static TfmbsTransferOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value src, ::mlir::Value dst, ::mlir::Value len);
  static TfmbsTransferOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value src, ::mlir::Value dst, ::mlir::Value len);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, ::mlir::Value len);
  static TfmbsTransferOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, ::mlir::Value len);
  static TfmbsTransferOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, ::mlir::Value len);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TfmbsTransferOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TfmbsTransferOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TfmbsTransferOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TfmbsTransferOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::tfmbs
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfmbs::TfmbsTransferOp)


#endif // GET_OP_CLASSES


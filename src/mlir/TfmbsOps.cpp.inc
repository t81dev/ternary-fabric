/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TfmbsOps.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::tfmbs::TfmbsDmaLoadOp,
::mlir::tfmbs::TfmbsFusedGemvOp,
::mlir::tfmbs::TfmbsGemvOp,
::mlir::tfmbs::TfmbsPackOp,
::mlir::tfmbs::TfmbsTransferOp
#endif // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TfmbsOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TfmbsOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of any type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TfmbsOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TfmbsOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TfmbsOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TfmbsOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::ArrayAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: array attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TfmbsOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TfmbsOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TfmbsOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::DictionaryAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: dictionary of named attribute values";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TfmbsOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TfmbsOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
namespace mlir::tfmbs {


//===----------------------------------------------------------------------===//
// ::mlir::tfmbs::TfmbsDmaLoadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

TfmbsDmaLoadOpGenericAdaptorBase::TfmbsDmaLoadOpGenericAdaptorBase(TfmbsDmaLoadOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
TfmbsDmaLoadOpAdaptor::TfmbsDmaLoadOpAdaptor(TfmbsDmaLoadOp op) : TfmbsDmaLoadOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TfmbsDmaLoadOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult TfmbsDmaLoadOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute TfmbsDmaLoadOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TfmbsDmaLoadOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> TfmbsDmaLoadOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void TfmbsDmaLoadOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void TfmbsDmaLoadOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult TfmbsDmaLoadOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void TfmbsDmaLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value host, ::mlir::Value fabric) {
  odsState.addOperands(host);
  odsState.addOperands(fabric);
}

TfmbsDmaLoadOp TfmbsDmaLoadOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value host, ::mlir::Value fabric) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(host)>(host), std::forward<decltype(fabric)>(fabric));
  auto __res__ = ::llvm::dyn_cast<TfmbsDmaLoadOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsDmaLoadOp TfmbsDmaLoadOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value host, ::mlir::Value fabric) {
  return create(builder, builder.getLoc(), std::forward<decltype(host)>(host), std::forward<decltype(fabric)>(fabric));
}

void TfmbsDmaLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value host, ::mlir::Value fabric) {
  odsState.addOperands(host);
  odsState.addOperands(fabric);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

TfmbsDmaLoadOp TfmbsDmaLoadOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value host, ::mlir::Value fabric) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(host)>(host), std::forward<decltype(fabric)>(fabric));
  auto __res__ = ::llvm::dyn_cast<TfmbsDmaLoadOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsDmaLoadOp TfmbsDmaLoadOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value host, ::mlir::Value fabric) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(host)>(host), std::forward<decltype(fabric)>(fabric));
}

void TfmbsDmaLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TfmbsDmaLoadOp TfmbsDmaLoadOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<TfmbsDmaLoadOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsDmaLoadOp TfmbsDmaLoadOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void TfmbsDmaLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TfmbsDmaLoadOp TfmbsDmaLoadOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<TfmbsDmaLoadOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsDmaLoadOp TfmbsDmaLoadOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult TfmbsDmaLoadOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TfmbsOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TfmbsOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TfmbsDmaLoadOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TfmbsDmaLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand hostRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> hostOperands(&hostRawOperand, 1);  ::llvm::SMLoc hostOperandsLoc;
  (void)hostOperandsLoc;
  ::mlir::Type hostRawType{};
  ::llvm::ArrayRef<::mlir::Type> hostTypes(&hostRawType, 1);
  ::mlir::OpAsmParser::UnresolvedOperand fabricRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> fabricOperands(&fabricRawOperand, 1);  ::llvm::SMLoc fabricOperandsLoc;
  (void)fabricOperandsLoc;
  ::mlir::Type fabricRawType{};
  ::llvm::ArrayRef<::mlir::Type> fabricTypes(&fabricRawType, 1);

  hostOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(hostRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    hostRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  fabricOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(fabricRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    fabricRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.resolveOperands(hostOperands, hostTypes, hostOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(fabricOperands, fabricTypes, fabricOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TfmbsDmaLoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getHost();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getHost().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getFabric();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getFabric().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}


} // namespace mlir::tfmbs
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfmbs::TfmbsDmaLoadOp)

namespace mlir::tfmbs {


//===----------------------------------------------------------------------===//
// ::mlir::tfmbs::TfmbsFusedGemvOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

TfmbsFusedGemvOpGenericAdaptorBase::TfmbsFusedGemvOpGenericAdaptorBase(TfmbsFusedGemvOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> TfmbsFusedGemvOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::IntegerAttr TfmbsFusedGemvOpGenericAdaptorBase::getTileMaskAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().tile_mask);
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0);
  return attr;
}

uint64_t TfmbsFusedGemvOpGenericAdaptorBase::getTileMask() {
  auto attr = getTileMaskAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::ArrayAttr > TfmbsFusedGemvOpGenericAdaptorBase::getTelemetry() {
  auto attr = getTelemetryAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}


} // namespace detail
TfmbsFusedGemvOpAdaptor::TfmbsFusedGemvOpAdaptor(TfmbsFusedGemvOp op) : TfmbsFusedGemvOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TfmbsFusedGemvOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_telemetry = getProperties().telemetry; (void)tblgen_telemetry;
  auto tblgen_tile_mask = getProperties().tile_mask; (void)tblgen_tile_mask;

  if (tblgen_tile_mask && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_tile_mask))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_tile_mask).getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfmbs.fused_gemv' op attribute 'tile_mask' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_telemetry && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_telemetry))))
    return emitError(loc, "'tfmbs.fused_gemv' op attribute 'telemetry' failed to satisfy constraint: array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TfmbsFusedGemvOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange TfmbsFusedGemvOp::getOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult TfmbsFusedGemvOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.telemetry;
       auto attr = dict.get("telemetry");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `telemetry` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.tile_mask;
       auto attr = dict.get("tile_mask");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `tile_mask` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TfmbsFusedGemvOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.telemetry;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("telemetry",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.tile_mask;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("tile_mask",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TfmbsFusedGemvOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.telemetry.getAsOpaquePointer()), 
    llvm::hash_value(prop.tile_mask.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TfmbsFusedGemvOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "telemetry")
      return prop.telemetry;

    if (name == "tile_mask")
      return prop.tile_mask;
  return std::nullopt;
}

void TfmbsFusedGemvOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "telemetry") {
       prop.telemetry = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.telemetry)>>(value);
       return;
    }

    if (name == "tile_mask") {
       prop.tile_mask = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.tile_mask)>>(value);
       return;
    }
}

void TfmbsFusedGemvOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.telemetry) attrs.append("telemetry", prop.telemetry);

    if (prop.tile_mask) attrs.append("tile_mask", prop.tile_mask);
}

::llvm::LogicalResult TfmbsFusedGemvOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTelemetryAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TfmbsOps2(attr, "telemetry", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTileMaskAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TfmbsOps1(attr, "tile_mask", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TfmbsFusedGemvOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.telemetry)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.tile_mask)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TfmbsFusedGemvOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.telemetry);

  writer.writeOptionalAttribute(prop.tile_mask);
}

uint64_t TfmbsFusedGemvOp::getTileMask() {
  auto attr = getTileMaskAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::ArrayAttr > TfmbsFusedGemvOp::getTelemetry() {
  auto attr = getTelemetryAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void TfmbsFusedGemvOp::setTileMask(::std::optional<uint64_t> attrValue) {
    auto &odsProp = getProperties().tile_mask;
    if (attrValue)
      odsProp = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue);
    else
      odsProp = nullptr;
}

void TfmbsFusedGemvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::ArrayAttr telemetry) {
  odsState.addOperands(operands);
  if (tile_mask) {
    odsState.getOrAddProperties<Properties>().tile_mask = tile_mask;
  }
  if (telemetry) {
    odsState.getOrAddProperties<Properties>().telemetry = telemetry;
  }
}

TfmbsFusedGemvOp TfmbsFusedGemvOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::ArrayAttr telemetry) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(operands)>(operands), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
  auto __res__ = ::llvm::dyn_cast<TfmbsFusedGemvOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsFusedGemvOp TfmbsFusedGemvOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::ArrayAttr telemetry) {
  return create(builder, builder.getLoc(), std::forward<decltype(operands)>(operands), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
}

void TfmbsFusedGemvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::ArrayAttr telemetry) {
  odsState.addOperands(operands);
  if (tile_mask) {
    odsState.getOrAddProperties<Properties>().tile_mask = tile_mask;
  }
  if (telemetry) {
    odsState.getOrAddProperties<Properties>().telemetry = telemetry;
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

TfmbsFusedGemvOp TfmbsFusedGemvOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::ArrayAttr telemetry) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
  auto __res__ = ::llvm::dyn_cast<TfmbsFusedGemvOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsFusedGemvOp TfmbsFusedGemvOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::ArrayAttr telemetry) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
}

void TfmbsFusedGemvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, /*optional*/uint64_t tile_mask, /*optional*/::mlir::ArrayAttr telemetry) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().tile_mask = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), tile_mask);
  if (telemetry) {
    odsState.getOrAddProperties<Properties>().telemetry = telemetry;
  }
}

TfmbsFusedGemvOp TfmbsFusedGemvOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, /*optional*/uint64_t tile_mask, /*optional*/::mlir::ArrayAttr telemetry) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(operands)>(operands), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
  auto __res__ = ::llvm::dyn_cast<TfmbsFusedGemvOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsFusedGemvOp TfmbsFusedGemvOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, /*optional*/uint64_t tile_mask, /*optional*/::mlir::ArrayAttr telemetry) {
  return create(builder, builder.getLoc(), std::forward<decltype(operands)>(operands), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
}

void TfmbsFusedGemvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, /*optional*/uint64_t tile_mask, /*optional*/::mlir::ArrayAttr telemetry) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().tile_mask = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), tile_mask);
  if (telemetry) {
    odsState.getOrAddProperties<Properties>().telemetry = telemetry;
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

TfmbsFusedGemvOp TfmbsFusedGemvOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, /*optional*/uint64_t tile_mask, /*optional*/::mlir::ArrayAttr telemetry) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
  auto __res__ = ::llvm::dyn_cast<TfmbsFusedGemvOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsFusedGemvOp TfmbsFusedGemvOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, /*optional*/uint64_t tile_mask, /*optional*/::mlir::ArrayAttr telemetry) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
}

void TfmbsFusedGemvOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TfmbsFusedGemvOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

TfmbsFusedGemvOp TfmbsFusedGemvOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<TfmbsFusedGemvOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsFusedGemvOp TfmbsFusedGemvOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void TfmbsFusedGemvOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TfmbsFusedGemvOp TfmbsFusedGemvOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<TfmbsFusedGemvOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsFusedGemvOp TfmbsFusedGemvOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult TfmbsFusedGemvOp::verifyInvariantsImpl() {
  auto tblgen_telemetry = getProperties().telemetry; (void)tblgen_telemetry;
  auto tblgen_tile_mask = getProperties().tile_mask; (void)tblgen_tile_mask;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TfmbsOps1(*this, tblgen_tile_mask, "tile_mask")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TfmbsOps2(*this, tblgen_telemetry, "telemetry")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TfmbsOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TfmbsFusedGemvOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult TfmbsFusedGemvOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;
  {

    auto &propStorage = prop.tile_mask;
    auto tile_maskAttrName = ::mlir::StringAttr::get(ctx, "tile_mask");
    auto attr = dict.get(tile_maskAttrName);
    usedKeys.insert(tile_maskAttrName);
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        emitError() << "expected key entry for tile_mask in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `tile_mask` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  {

    auto &propStorage = prop.telemetry;
    auto telemetryAttrName = ::mlir::StringAttr::get(ctx, "telemetry");
    auto attr = dict.get(telemetryAttrName);
    usedKeys.insert(telemetryAttrName);
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        emitError() << "expected key entry for telemetry in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `telemetry` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}


} // namespace mlir::tfmbs
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfmbs::TfmbsFusedGemvOp)

namespace mlir::tfmbs {


//===----------------------------------------------------------------------===//
// ::mlir::tfmbs::TfmbsGemvOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

TfmbsGemvOpGenericAdaptorBase::TfmbsGemvOpGenericAdaptorBase(TfmbsGemvOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::IntegerAttr TfmbsGemvOpGenericAdaptorBase::getTileMaskAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::IntegerAttr>(getProperties().tile_mask);
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0);
  return attr;
}

uint64_t TfmbsGemvOpGenericAdaptorBase::getTileMask() {
  auto attr = getTileMaskAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::DictionaryAttr > TfmbsGemvOpGenericAdaptorBase::getTelemetry() {
  auto attr = getTelemetryAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}


} // namespace detail
TfmbsGemvOpAdaptor::TfmbsGemvOpAdaptor(TfmbsGemvOp op) : TfmbsGemvOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TfmbsGemvOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_telemetry = getProperties().telemetry; (void)tblgen_telemetry;
  auto tblgen_tile_mask = getProperties().tile_mask; (void)tblgen_tile_mask;

  if (tblgen_tile_mask && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_tile_mask))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_tile_mask).getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfmbs.gemv' op attribute 'tile_mask' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_telemetry && !((::llvm::isa<::mlir::DictionaryAttr>(tblgen_telemetry))))
    return emitError(loc, "'tfmbs.gemv' op attribute 'telemetry' failed to satisfy constraint: dictionary of named attribute values");
  return ::mlir::success();
}

::llvm::LogicalResult TfmbsGemvOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.telemetry;
       auto attr = dict.get("telemetry");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `telemetry` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.tile_mask;
       auto attr = dict.get("tile_mask");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `tile_mask` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TfmbsGemvOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.telemetry;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("telemetry",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.tile_mask;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("tile_mask",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TfmbsGemvOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.telemetry.getAsOpaquePointer()), 
    llvm::hash_value(prop.tile_mask.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TfmbsGemvOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "telemetry")
      return prop.telemetry;

    if (name == "tile_mask")
      return prop.tile_mask;
  return std::nullopt;
}

void TfmbsGemvOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "telemetry") {
       prop.telemetry = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.telemetry)>>(value);
       return;
    }

    if (name == "tile_mask") {
       prop.tile_mask = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.tile_mask)>>(value);
       return;
    }
}

void TfmbsGemvOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.telemetry) attrs.append("telemetry", prop.telemetry);

    if (prop.tile_mask) attrs.append("tile_mask", prop.tile_mask);
}

::llvm::LogicalResult TfmbsGemvOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTelemetryAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TfmbsOps3(attr, "telemetry", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getTileMaskAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TfmbsOps1(attr, "tile_mask", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult TfmbsGemvOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.telemetry)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.tile_mask)))
    return ::mlir::failure();
  return ::mlir::success();
}

void TfmbsGemvOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.telemetry);

  writer.writeOptionalAttribute(prop.tile_mask);
}

uint64_t TfmbsGemvOp::getTileMask() {
  auto attr = getTileMaskAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::std::optional< ::mlir::DictionaryAttr > TfmbsGemvOp::getTelemetry() {
  auto attr = getTelemetryAttr();
  return attr ? ::std::optional< ::mlir::DictionaryAttr >(attr) : (::std::nullopt);
}

void TfmbsGemvOp::setTileMask(::std::optional<uint64_t> attrValue) {
    auto &odsProp = getProperties().tile_mask;
    if (attrValue)
      odsProp = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue);
    else
      odsProp = nullptr;
}

void TfmbsGemvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::DictionaryAttr telemetry) {
  odsState.addOperands(weight);
  odsState.addOperands(input);
  odsState.addOperands(output);
  if (tile_mask) {
    odsState.getOrAddProperties<Properties>().tile_mask = tile_mask;
  }
  if (telemetry) {
    odsState.getOrAddProperties<Properties>().telemetry = telemetry;
  }
}

TfmbsGemvOp TfmbsGemvOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::DictionaryAttr telemetry) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(weight)>(weight), std::forward<decltype(input)>(input), std::forward<decltype(output)>(output), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
  auto __res__ = ::llvm::dyn_cast<TfmbsGemvOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsGemvOp TfmbsGemvOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::DictionaryAttr telemetry) {
  return create(builder, builder.getLoc(), std::forward<decltype(weight)>(weight), std::forward<decltype(input)>(input), std::forward<decltype(output)>(output), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
}

void TfmbsGemvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::DictionaryAttr telemetry) {
  odsState.addOperands(weight);
  odsState.addOperands(input);
  odsState.addOperands(output);
  if (tile_mask) {
    odsState.getOrAddProperties<Properties>().tile_mask = tile_mask;
  }
  if (telemetry) {
    odsState.getOrAddProperties<Properties>().telemetry = telemetry;
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

TfmbsGemvOp TfmbsGemvOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::DictionaryAttr telemetry) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(weight)>(weight), std::forward<decltype(input)>(input), std::forward<decltype(output)>(output), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
  auto __res__ = ::llvm::dyn_cast<TfmbsGemvOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsGemvOp TfmbsGemvOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::IntegerAttr tile_mask, /*optional*/::mlir::DictionaryAttr telemetry) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(weight)>(weight), std::forward<decltype(input)>(input), std::forward<decltype(output)>(output), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
}

void TfmbsGemvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/uint64_t tile_mask, /*optional*/::mlir::DictionaryAttr telemetry) {
  odsState.addOperands(weight);
  odsState.addOperands(input);
  odsState.addOperands(output);
  odsState.getOrAddProperties<Properties>().tile_mask = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), tile_mask);
  if (telemetry) {
    odsState.getOrAddProperties<Properties>().telemetry = telemetry;
  }
}

TfmbsGemvOp TfmbsGemvOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/uint64_t tile_mask, /*optional*/::mlir::DictionaryAttr telemetry) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(weight)>(weight), std::forward<decltype(input)>(input), std::forward<decltype(output)>(output), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
  auto __res__ = ::llvm::dyn_cast<TfmbsGemvOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsGemvOp TfmbsGemvOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/uint64_t tile_mask, /*optional*/::mlir::DictionaryAttr telemetry) {
  return create(builder, builder.getLoc(), std::forward<decltype(weight)>(weight), std::forward<decltype(input)>(input), std::forward<decltype(output)>(output), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
}

void TfmbsGemvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/uint64_t tile_mask, /*optional*/::mlir::DictionaryAttr telemetry) {
  odsState.addOperands(weight);
  odsState.addOperands(input);
  odsState.addOperands(output);
  odsState.getOrAddProperties<Properties>().tile_mask = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), tile_mask);
  if (telemetry) {
    odsState.getOrAddProperties<Properties>().telemetry = telemetry;
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

TfmbsGemvOp TfmbsGemvOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/uint64_t tile_mask, /*optional*/::mlir::DictionaryAttr telemetry) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(weight)>(weight), std::forward<decltype(input)>(input), std::forward<decltype(output)>(output), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
  auto __res__ = ::llvm::dyn_cast<TfmbsGemvOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsGemvOp TfmbsGemvOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value weight, ::mlir::Value input, ::mlir::Value output, /*optional*/uint64_t tile_mask, /*optional*/::mlir::DictionaryAttr telemetry) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(weight)>(weight), std::forward<decltype(input)>(input), std::forward<decltype(output)>(output), std::forward<decltype(tile_mask)>(tile_mask), std::forward<decltype(telemetry)>(telemetry));
}

void TfmbsGemvOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<TfmbsGemvOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

TfmbsGemvOp TfmbsGemvOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<TfmbsGemvOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsGemvOp TfmbsGemvOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void TfmbsGemvOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TfmbsGemvOp TfmbsGemvOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<TfmbsGemvOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsGemvOp TfmbsGemvOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult TfmbsGemvOp::verifyInvariantsImpl() {
  auto tblgen_telemetry = getProperties().telemetry; (void)tblgen_telemetry;
  auto tblgen_tile_mask = getProperties().tile_mask; (void)tblgen_tile_mask;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TfmbsOps1(*this, tblgen_tile_mask, "tile_mask")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TfmbsOps3(*this, tblgen_telemetry, "telemetry")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TfmbsOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TfmbsOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TfmbsOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TfmbsGemvOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TfmbsGemvOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand weightRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> weightOperands(&weightRawOperand, 1);  ::llvm::SMLoc weightOperandsLoc;
  (void)weightOperandsLoc;
  ::mlir::Type weightRawType{};
  ::llvm::ArrayRef<::mlir::Type> weightTypes(&weightRawType, 1);
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);
  ::mlir::OpAsmParser::UnresolvedOperand outputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> outputOperands(&outputRawOperand, 1);  ::llvm::SMLoc outputOperandsLoc;
  (void)outputOperandsLoc;
  ::mlir::Type outputRawType{};
  ::llvm::ArrayRef<::mlir::Type> outputTypes(&outputRawType, 1);

  weightOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(weightRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    weightRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  outputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(outputRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    outputRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.resolveOperands(weightOperands, weightTypes, weightOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(outputOperands, outputTypes, outputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TfmbsGemvOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getWeight();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getWeight().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getOutput();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOutput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getTileMaskAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0)))
       elidedAttrs.push_back("tile_mask");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}


} // namespace mlir::tfmbs
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfmbs::TfmbsGemvOp)

namespace mlir::tfmbs {


//===----------------------------------------------------------------------===//
// ::mlir::tfmbs::TfmbsPackOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

TfmbsPackOpGenericAdaptorBase::TfmbsPackOpGenericAdaptorBase(TfmbsPackOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
TfmbsPackOpAdaptor::TfmbsPackOpAdaptor(TfmbsPackOp op) : TfmbsPackOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TfmbsPackOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult TfmbsPackOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute TfmbsPackOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TfmbsPackOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> TfmbsPackOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void TfmbsPackOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void TfmbsPackOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult TfmbsPackOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void TfmbsPackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dst) {
  odsState.addOperands(src);
  odsState.addOperands(dst);
}

TfmbsPackOp TfmbsPackOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value src, ::mlir::Value dst) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(src)>(src), std::forward<decltype(dst)>(dst));
  auto __res__ = ::llvm::dyn_cast<TfmbsPackOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsPackOp TfmbsPackOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value src, ::mlir::Value dst) {
  return create(builder, builder.getLoc(), std::forward<decltype(src)>(src), std::forward<decltype(dst)>(dst));
}

void TfmbsPackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst) {
  odsState.addOperands(src);
  odsState.addOperands(dst);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

TfmbsPackOp TfmbsPackOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src)>(src), std::forward<decltype(dst)>(dst));
  auto __res__ = ::llvm::dyn_cast<TfmbsPackOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsPackOp TfmbsPackOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src)>(src), std::forward<decltype(dst)>(dst));
}

void TfmbsPackOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TfmbsPackOp TfmbsPackOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<TfmbsPackOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsPackOp TfmbsPackOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void TfmbsPackOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TfmbsPackOp TfmbsPackOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<TfmbsPackOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsPackOp TfmbsPackOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult TfmbsPackOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TfmbsOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TfmbsOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TfmbsPackOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TfmbsPackOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::Type srcRawType{};
  ::llvm::ArrayRef<::mlir::Type> srcTypes(&srcRawType, 1);
  ::mlir::OpAsmParser::UnresolvedOperand dstRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dstOperands(&dstRawOperand, 1);  ::llvm::SMLoc dstOperandsLoc;
  (void)dstOperandsLoc;
  ::mlir::Type dstRawType{};
  ::llvm::ArrayRef<::mlir::Type> dstTypes(&dstRawType, 1);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    srcRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  dstOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dstRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    dstRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dstOperands, dstTypes, dstOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TfmbsPackOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDst();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getDst().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}


} // namespace mlir::tfmbs
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfmbs::TfmbsPackOp)

namespace mlir::tfmbs {


//===----------------------------------------------------------------------===//
// ::mlir::tfmbs::TfmbsTransferOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

TfmbsTransferOpGenericAdaptorBase::TfmbsTransferOpGenericAdaptorBase(TfmbsTransferOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
TfmbsTransferOpAdaptor::TfmbsTransferOpAdaptor(TfmbsTransferOp op) : TfmbsTransferOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TfmbsTransferOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult TfmbsTransferOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute TfmbsTransferOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TfmbsTransferOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> TfmbsTransferOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void TfmbsTransferOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void TfmbsTransferOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult TfmbsTransferOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void TfmbsTransferOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dst, ::mlir::Value len) {
  odsState.addOperands(src);
  odsState.addOperands(dst);
  odsState.addOperands(len);
}

TfmbsTransferOp TfmbsTransferOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value src, ::mlir::Value dst, ::mlir::Value len) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(src)>(src), std::forward<decltype(dst)>(dst), std::forward<decltype(len)>(len));
  auto __res__ = ::llvm::dyn_cast<TfmbsTransferOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsTransferOp TfmbsTransferOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value src, ::mlir::Value dst, ::mlir::Value len) {
  return create(builder, builder.getLoc(), std::forward<decltype(src)>(src), std::forward<decltype(dst)>(dst), std::forward<decltype(len)>(len));
}

void TfmbsTransferOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, ::mlir::Value len) {
  odsState.addOperands(src);
  odsState.addOperands(dst);
  odsState.addOperands(len);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

TfmbsTransferOp TfmbsTransferOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, ::mlir::Value len) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src)>(src), std::forward<decltype(dst)>(dst), std::forward<decltype(len)>(len));
  auto __res__ = ::llvm::dyn_cast<TfmbsTransferOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsTransferOp TfmbsTransferOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, ::mlir::Value len) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(src)>(src), std::forward<decltype(dst)>(dst), std::forward<decltype(len)>(len));
}

void TfmbsTransferOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TfmbsTransferOp TfmbsTransferOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<TfmbsTransferOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsTransferOp TfmbsTransferOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void TfmbsTransferOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TfmbsTransferOp TfmbsTransferOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<TfmbsTransferOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TfmbsTransferOp TfmbsTransferOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult TfmbsTransferOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TfmbsOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TfmbsOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TfmbsOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TfmbsTransferOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TfmbsTransferOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(&srcRawOperand, 1);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::Type srcRawType{};
  ::llvm::ArrayRef<::mlir::Type> srcTypes(&srcRawType, 1);
  ::mlir::OpAsmParser::UnresolvedOperand dstRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dstOperands(&dstRawOperand, 1);  ::llvm::SMLoc dstOperandsLoc;
  (void)dstOperandsLoc;
  ::mlir::Type dstRawType{};
  ::llvm::ArrayRef<::mlir::Type> dstTypes(&dstRawType, 1);
  ::mlir::OpAsmParser::UnresolvedOperand lenRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lenOperands(&lenRawOperand, 1);  ::llvm::SMLoc lenOperandsLoc;
  (void)lenOperandsLoc;
  ::mlir::Type lenRawType{};
  ::llvm::ArrayRef<::mlir::Type> lenTypes(&lenRawType, 1);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    srcRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  dstOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dstRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    dstRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  lenOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lenRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lenRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dstOperands, dstTypes, dstOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(lenOperands, lenTypes, lenOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TfmbsTransferOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDst();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getDst().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getLen();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getLen().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}


} // namespace mlir::tfmbs
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfmbs::TfmbsTransferOp)


#endif // GET_OP_CLASSES


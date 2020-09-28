import 'package:flutter/foundation.dart';
import 'package:pytorch_flutter/ivalue.dart';

/// The available data types for the input
///
/// Internally during native call, they will be converted to Tensor if required
/// this behaviour is defined in the [IValue] class
enum ValueType {
  BOOL,
  INT,
  FLOAT,
  STR
}

/// Gets the string representation of the enum [DType]
///
/// DType.float32.inString => 'float32'
extension ValueTypeToString on ValueType {
  String get inString => describeEnum(this);
}

enum TensorType {
  UINT8,
  INT8,
  FLOAT32,
  INT32,
  FLOAT64,
  INT64,
}

extension TensorTypeToString on TensorType {
  String get inString => describeEnum(this);
}
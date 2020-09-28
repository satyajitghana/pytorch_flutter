import 'dart:typed_data';

import 'package:flutter/cupertino.dart';
import 'package:pytorch_flutter/dtypes.dart';


/// The class representing the data, data type and shape of the tensor
/// which is going to be fed to the model's [forward] function
abstract class IValue {
  var data;

  /// Constructs an IValue object
  ///
  /// Takes in [data] at argument, which is a list containing the input values
  /// all the values will be converted to type [dtype] Tensor during calland the
  /// shape for the tensor is defined as a list in [shape] argument
  IValue(this.data);

  /// Converts the IValue into a [Map] which will be used by [invokeMethod]
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      "class_type": this.runtimeType.toString(),
      "data": data,
    };
  }
}

/// [Value] represents the following
///
/// Dart -> Java
/// bool -> Boolean
/// int -> Integer, Long
/// double -> Double
/// String -> String
///
/// These CANNOT be set to Tensors
///
/// The given [data] will be converted to [dtype] at the Backend
class Value extends IValue {
  // data type to convert to at Backend
  ValueType dtype;

  Value({ @required data, @required this.dtype })
      : assert(
  ['bool', 'int', 'double', 'String'].any((e) => e == data.runtimeType
      .toString()), "Found data of type: ${data.runtimeType}: Only data of type bool, int, double, String is allowed"
  ),
        super(data);

  // returns a representation of this class
  @override
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      ...super.toMap(),
      "dtype": dtype.inString,
    };
  }
}

/// [TensorValue] represents the following
///
/// Dart -> Java
/// Uint8List -> byte[]
/// Int32List -> int[]
/// Int64List -> long[]
/// Float64List -> double[]
class TensorValue extends IValue {
  TensorType dtype;
  List<int> shape;

  TensorValue({ @required data, @required this.shape, @required this.dtype }) : assert(
  ['Uint8List', 'Int32List', 'Int64List', 'Float64List'].any((e) => e == data.runtimeType.toString()),
  "Found data of type: ${data.runtimeType}: Only data of type Uint8List, Int32List, Int64List, Float64List is allowed"
  ), super(data);

  // returns a representation of this class
  @override
  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      ...super.toMap(),
      "shape" : Int64List.fromList(shape),
      "dtype": dtype.inString,
    };
  }

}

/// [ImageTensor] represents an Image
/// It can only contain a type Uint8List
class ImageTensor extends IValue {
  // The image will be normalized using these values
  List<double> mean = [0.485, 0.456, 0.406];
  List<double> std = [0.229, 0.224, 0.225];

  ImageTensor(
      { @required Uint8List data, this.mean, this.std })
      : super(data);

  // returns a representation of this class
  @override
  Map<String, dynamic> toMap() {
    // Float64List will be converted to double[] in Java,
    // then we can use the bitmapToFloat32Tensor to convert the image to tensor
    return <String, dynamic>{
      ...super.toMap(),
      "mean": Float64List.fromList(mean),
      "std": Float64List.fromList(std),
    };
  }
}

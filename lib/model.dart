import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:pytorch_flutter/dtypes.dart';
import 'package:pytorch_flutter/ivalue.dart';

class Model {
  // the method channel defined in the native code
  static const MethodChannel _channel = const MethodChannel('pytorch_flutter');

  /// the model _index, a private property, that keeps track of the index in the
  /// array in native code
  final int _index;

  Model._internal(this._index);

  /// Loads a Model from given [assetPath]
  static Future<Model> fromAsset({String assetPath}) async {
    int index = await _channel.invokeMethod("loadModelFromAsset");
    return Model._internal(index);
  }

  /// Loads a Model from given [filePath]
  static Future<Model> fromFile({String filePath}) async {
    int index = await _channel.invokeMethod("loadModelFromFile");
    return Model._internal(index);
  }

  /// Runs the forward function of the model with the input given
  ///
  /// Takes in [input] which is an [IValue] and passes it to the forward
  /// 'function' of the torch::jit model
  ///
  /// Sample usage
  /// ```
  /// ptfl.Model meow = ptfl.Model.fromAsset('assets/yolov4.traced.pt')
  /// meow.forward(input)
  /// ```
  Future<dynamic> forward(IValue input) async {
    final output =
        await _channel.invokeMethod("forward", {
          "index": _index,
          ...input.toMap()
        });

    return output;
  }

  /// Runs the [inputs] parameters on the model
  ///
  /// This is the same as the [forward] function, but [forwardList] is used when
  /// your model's forward function has multiple inputs like
  /// ```
  /// def forward(self, input1, input2, input3)
  ///   x = self.main(input1)
  /// ```
  /// In the above case, use this function with the appropriate list of values
  Future<dynamic> forwardList(List<IValue> inputs) async {
    final output =
        await _channel.invokeMethod('forwardList', <String, dynamic>{
          "index": _index,
          "inputs": inputs.map((e) => e.toMap()).toList()
        });

    return output;
  }

  /// Runs the [methodName] of the Module with input param as [input]
  Future<dynamic> runMethod(String methodName, IValue input) async {
    final output =
        await _channel.invokeMethod("runMethod", <String, dynamic>{
          "index": _index,
          "methodName": methodName,
          ...input.toMap(),
        });

    return output;
  }

  /// Runs the [methodName] of the Module with varargs inputs [inputs]
  Future<dynamic> runMethodList(String methodName, List<IValue> inputs) async {
    final output =
    await _channel.invokeMethod('runMethodList', <String, dynamic>{
      "index": _index,
      "methodName": methodName,
      "inputs": inputs.map((e) => e.toMap()).toList()
    });

    return output;
  }

  /// Disposes and frees the resources occupied by the [Model]
  ///
  /// When you are done with the model and would like to free up the memory
  /// occupied by it, call dispose. It's important, since garbage collection
  /// on the Native side won't take place unless dispose is called from the
  /// Flutter end
  ///
  /// NOTE: Once dispose is called, the model object should not be used anymore
  Future<bool> dispose() async {
    return await _channel
        .invokeMethod('closeModel', <String, int>{"_index": _index});
  }

//  /// predicts abstract number input
//  Future<List> getPrediction(
//      List<double> input, List<int> shape, DType dtype, DType oDtype) async {
//    final List prediction = await _channel.invokeListMethod('predict', {
//      "index": _index,
//      "data": input,
//      "shape": shape,
//      "dtype": dtype.toString().split(".").last,
//      "oDtype": oDtype.toString().split(".").last,
//    });
//    return prediction;
//  }
//
//  Future<List<List>> getPredictionList(
//      List<double> input, List<int> shape, DType dtype, DType oDtype) async {
//    final List<List> prediction = await _channel.invokeListMethod('predict', {
//      "index": _index,
//      "data": input,
//      "shape": shape,
//      "dtype": dtype.toString().split(".").last,
//      "oDtype": oDtype.toString().split(".").last,
//    });
//    return prediction;
//  }
}

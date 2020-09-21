import 'dart:io';

import 'package:flutter/services.dart';

enum DType {
  float32,
  float64,
  int32,
  int64,
  int8,
  uint8,
}

class Model {
  static const MethodChannel _channel = const MethodChannel('pytorch_flutter');

  final int _index;

  Model(this._index);

  /// predicts abstract number input
  Future<List> getPrediction(
      List<double> input, List<int> shape, DType dtype, DType oDtype) async {
    final List prediction = await _channel.invokeListMethod('predict', {
      "index": _index,
      "data": input,
      "shape": shape,
      "dtype": dtype.toString().split(".").last,
      "oDtype": oDtype.toString().split(".").last,
    });
    return prediction;
  }

  Future<List<List>> getPredictionList(
      List<double> input, List<int> shape, DType dtype, DType oDtype) async {
    final List<List> prediction = await _channel.invokeListMethod('predict', {
      "index": _index,
      "data": input,
      "shape": shape,
      "dtype": dtype.toString().split(".").last,
      "oDtype": oDtype.toString().split(".").last,
    });
    return prediction;
  }
}
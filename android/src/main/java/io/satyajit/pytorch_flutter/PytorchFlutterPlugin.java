package io.satyajit.pytorch_flutter;

import androidx.annotation.NonNull;

import android.util.Log;

import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;
import io.flutter.plugin.common.PluginRegistry.Registrar;

import org.pytorch.DType;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.lang.reflect.Array;
import java.util.ArrayList;

public class PytorchFlutterPlugin implements FlutterPlugin, MethodCallHandler {
  static String LOGTAG = "PyTorchFlutter";
  private MethodChannel channel;
  ArrayList<Module> modules = new ArrayList<>();

  @Override
  public void onAttachedToEngine(@NonNull FlutterPluginBinding flutterPluginBinding) {
    channel = new MethodChannel(flutterPluginBinding.getBinaryMessenger(), "pytorch_flutter");
    channel.setMethodCallHandler(this);
  }

  @Override
  public void onMethodCall(@NonNull MethodCall call, @NonNull Result result) {

    switch (call.method) {
      case "loadModel":
        try {
          String absPath = call.argument("absPath");
          modules.add(Module.load(absPath));
          result.success(modules.size() - 1);
          Log.i(LOGTAG, absPath + " model is loaded !");
        } catch (Exception e) {
          String assetPath = call.argument("assetPath");
          Log.e(LOGTAG, assetPath + " cannot be loaded: ", e);
        }
        break;
      case "predict":
        Module module = null;
        Integer[] shape = null;
        Double[] data = null;
        DType dtype = null;
        DType oDtype = null;
        try {
          int index = call.argument("index");
          module = modules.get(index);

          dtype = DType.valueOf(call.argument("dtype").toString().toUpperCase());
          oDtype = DType.valueOf(call.argument("oDtype").toString().toUpperCase());

          ArrayList<Integer> shapeList = call.argument("shape");
          shape = shapeList.toArray(new Integer[shapeList.size()]);

          ArrayList<Double> dataList = call.argument("data");
          data = dataList.toArray(new Double[dataList.size()]);
        } catch (Exception e) {
          Log.e(LOGTAG, "error parsing arguments: ", e);
        }

        final Tensor inputTensor = getInputTensor(dtype, data, shape);
        Log.e( LOGTAG, "DataType of Input Tensor: " + inputTensor.dtype().toString() );

//        Tensor outputTensor = null;
        IValue modelOutputValue = null;
        try {
//          outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
          modelOutputValue = module.forward(IValue.from(inputTensor));
        } catch(RuntimeException e){
          Log.e(LOGTAG, "Your input type " + dtype.toString().toLowerCase()  + " (" + Convert.dtypeAsPrimitive(dtype.toString()) +") " + "does not match with model input type", e);
          result.success(null);
        }

        successResult(result, oDtype, modelOutputValue);

        break;
      default:
        result.notImplemented();
        break;
    }

  }

  private Tensor getInputTensor(DType dtype, Double[] data, Integer[] shape){
    switch (dtype){
      case FLOAT32:
        return Tensor.fromBlob(Convert.toFloatPrimitives(data), Convert.toPrimitives(shape));
      case FLOAT64:
        return  Tensor.fromBlob(Convert.toPrimitives(shape), Convert.toDoublePrimitives(data));
      case INT32:
        return Tensor.fromBlob(Convert.toIntegerPrimitives(data), Convert.toPrimitives(shape));
      case INT64:
        return Tensor.fromBlob(Convert.toLongPrimitives(data), Convert.toPrimitives(shape));
      case INT8:
        return Tensor.fromBlob(Convert.toBytePrimitives(data), Convert.toPrimitives(shape));
      case UINT8:
        return Tensor.fromBlobUnsigned(Convert.toBytePrimitives(data), Convert.toPrimitives(shape));
      default:
        return null;
    }
  }

  /**
   * Returns the output from the model back to flutter caller and also converts
   * the output to the desired value
   *
   * @param result the flutter result handler object
   * @param dtype the output data type
   * @param outputValue the output as `IValue` from the model
   */
  private void successResult(Result result, DType dtype, IValue outputValue){

    if (outputValue.isTensor()) {
      Tensor outputTensor = outputValue.toTensor();
      switch (dtype) {
        case FLOAT32:
          ArrayList<Float> outputListFloat = new ArrayList<>();
          for (float f : outputTensor.getDataAsFloatArray()) {
            outputListFloat.add(f);
          }
          result.success(outputListFloat);
          break;
        case FLOAT64:
          ArrayList<Double> outputListDouble = new ArrayList<>();
          for (double d : outputTensor.getDataAsDoubleArray()) {
            outputListDouble.add(d);
          }
          result.success(outputListDouble);
          break;
        case INT32:
          ArrayList<Integer> outputListInteger = new ArrayList<>();
          for (int i : outputTensor.getDataAsIntArray()) {
            outputListInteger.add(i);
          }
          result.success(outputListInteger);
          break;
        case INT64:
          ArrayList<Long> outputListLong = new ArrayList<>();
          for (long l : outputTensor.getDataAsLongArray()) {
            outputListLong.add(l);
          }
          result.success(outputListLong);
          break;
        case INT8:
          ArrayList<Byte> outputListByte = new ArrayList<>();
          for (byte b : outputTensor.getDataAsByteArray()) {
            outputListByte.add(b);
          }
          result.success(outputListByte);
          break;
        case UINT8:
          ArrayList<Byte> outputListUByte = new ArrayList<>();
          for (byte ub : outputTensor.getDataAsUnsignedByteArray()) {
            outputListUByte.add(ub);
          }
          result.success(outputListUByte);
          break;
        default:
          result.success(null);
          break;
      }
    } else if (outputValue.isTensorList()) {
      Tensor[] outputTensorList = outputValue.toTensorList();
      switch (dtype) {
        case FLOAT32:
          ArrayList<ArrayList<Float>> outputListFloat = new ArrayList<>();
          for (Tensor t_ : outputTensorList) {
            ArrayList<Float> out_ = new ArrayList<>();
            for (float f : t_.getDataAsFloatArray()) {
              out_.add(f);
            }
            outputListFloat.add(out_);
          }
          result.success(outputListFloat);
          break;
        case FLOAT64:
          ArrayList<ArrayList<Double>> outputListDouble = new ArrayList<>();
          for (Tensor t_ : outputTensorList) {
            ArrayList<Double> out_ = new ArrayList<>();
            for (double d : t_.getDataAsDoubleArray()) {
              out_.add(d);
            }
            outputListDouble.add(out_);
          }
          result.success(outputListDouble);
          break;
        case INT32:
          ArrayList<ArrayList<Integer>> outputListInteger = new ArrayList<>();
          for (Tensor t_ : outputTensorList) {
            ArrayList<Integer> out_ = new ArrayList<>();
            for (int i : t_.getDataAsIntArray()) {
              out_.add(i);
            }
            outputListInteger.add(out_);
          }
          result.success(outputListInteger);
          break;
        case INT64:
          ArrayList<ArrayList<Long>> outputListLong = new ArrayList<>();
          for (Tensor t_ : outputTensorList) {
            ArrayList<Long> out_ = new ArrayList<>();
            for (long l : t_.getDataAsLongArray()) {
              out_.add(l);
            }
            outputListLong.add(out_);
          }
          result.success(outputListLong);
          break;
        case INT8:
          ArrayList<ArrayList<Byte>> outputListByte = new ArrayList<>();
          for (Tensor t_ : outputTensorList) {
            ArrayList<Byte> out_ = new ArrayList<>();
            for (byte b : t_.getDataAsByteArray()) {
              out_.add(b);
            }
            outputListByte.add(out_);
          }
          result.success(outputListByte);
          break;
        case UINT8:
          ArrayList<ArrayList<Byte>> outputListUByte = new ArrayList<>();
          for (Tensor t_ : outputTensorList) {
            ArrayList<Byte> out_ = new ArrayList<>();
            for (byte ub : t_.getDataAsUnsignedByteArray()) {
              out_.add(ub);
            }
            outputListUByte.add(out_);
          }
          result.success(outputListUByte);
          break;
        default:
          result.success(null);
          break;
      }
    }
  }

  @Override
  public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
    channel.setMethodCallHandler(null);
  }
}

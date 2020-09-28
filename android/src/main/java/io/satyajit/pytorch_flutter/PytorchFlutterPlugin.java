package io.satyajit.pytorch_flutter;

import androidx.annotation.NonNull;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import com.google.common.primitives.Bytes;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Floats;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;

import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;

import org.jetbrains.annotations.NotNull;
import org.pytorch.DType;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.security.spec.ECField;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.pytorch.torchvision.TensorImageUtils;


public class PytorchFlutterPlugin implements FlutterPlugin, MethodCallHandler {
    static String LOGTAG = "PyTorchFlutter";
    static String CHANNEL = "pytorch_flutter";

    enum PTFLErrors {
        ERROR_LOADINNG_MODEL,
        ERROR_RUNNING_FORWARD,
        ERROR_RUNNING_FORWARD_LIST,
        ERROR_DATATYPE_MISMATCH
    }

    // Flutter Binding Variables
    private MethodChannel channel;
    private Context applicationContext;
    private FlutterPluginBinding pluginBinding;

    // PyTorch Plugin Variables
    ArrayList<Module> modules = new ArrayList<>();

    @Override
    public void onAttachedToEngine(@NonNull FlutterPluginBinding flutterPluginBinding) {
        pluginBinding = flutterPluginBinding;

        channel = new MethodChannel(flutterPluginBinding.getBinaryMessenger(), CHANNEL);
        channel.setMethodCallHandler(this);

        applicationContext = flutterPluginBinding.getApplicationContext();
    }

    @Override
    public void onMethodCall(@NonNull MethodCall call, @NonNull Result result) {

        switch (call.method) {
            case "loadModelFromAsset": {
                try {
                    int modelIndex = loadModelFromAsset((HashMap) call.arguments);
                    result.success(modelIndex);
                } catch (Exception e) {
                    result.error(PTFLErrors.ERROR_LOADINNG_MODEL.toString(), "Error Loading model from asset", e);
                }
            }
            break;
            case "loadModelFromFile": {
                try {
                    int modelIndex = loadModelFromFile((HashMap) call.arguments);
                    result.success(modelIndex);
                } catch (Exception e) {
                    result.error(PTFLErrors.ERROR_LOADINNG_MODEL.toString(), "Error Loading model from file", e);
                }
            }
            break;
            /// "forward" is used to run the "forward" method of the module
            case "forward": {
                try {
                    Object output = modelForward(call);
                    result.success(output);
                } catch (Exception e) {
                    result.error(PTFLErrors.ERROR_RUNNING_FORWARD.toString(), "Error running forward(input) on Model", e);
                }
            }
            break;
            /// "forwardList" is used to run the "forward" method of the module but with variable list of
            /// arguments
            case "forwardList": {
                try {
                    Object output = modelForwardList(call);
                    result.success(output);
                } catch (Exception e) {
                    result.error(PTFLErrors.ERROR_RUNNING_FORWARD_LIST.toString(), "Error running forward(...inputs) on Model", e);
                }
            }
            break;
            case "runMethod": {
                try {
                    Object output = modelRunMethod(call);
                    result.success(output);
                } catch (Exception e) {
                    result.error(PTFLErrors.ERROR_RUNNING_FORWARD_LIST.toString(), "Error running forward(...inputs) on Model", e);
                }
            }
            break;
            case "runMethodList": {
                try {
                    Object output = modelRunMethodList(call);
                    result.success(output);
                } catch (Exception e) {
                    result.error(PTFLErrors.ERROR_RUNNING_FORWARD_LIST.toString(), "Error running forward(...inputs) on Model", e);
                }
            }
            break;
            default:
                result.notImplemented();
                break;
        }

    }

    /**
     * Runs a specific method of the Module with variable list of arguments
     *
     * @param call the call parameters from Flutter end
     * @return Object the output from the model
     * @throws Exception throws Exception if any error happens during the forward process or when
     *  decoding the inputs passed to the runMethod function
     */
    private Object modelRunMethodList(MethodCall call) throws Exception {
        int index = Objects.requireNonNull(call.argument("index"));
        String methodName = call.argument("methodName");

        Module module = modules.get(index);

        ArrayList<HashMap<?, ?>> inputsArg = call.argument("inputs");

        assert inputsArg != null;
        assert module != null;

        ArrayList<IValue> inputs = new ArrayList<>();
        // prepare the input arguments
        for (HashMap<?, ?> arg : inputsArg) {
            inputs.add(parseIValue(arg));
        }

        Object output = module.runMethod(methodName, inputs.toArray(new IValue[0]));

        return output;

    }

    /**
     * Run a specific method from the loaded Module
     *
     * @param call The call from Flutter end for this method
     * @return (Object) a representation of the output received from running the specified
     *  `runMethod` on the model.
     * @throws Exception if any any exception takes place while parsing the inputs or while running
     *  `runMethod` on the model.
     */
    private Object modelRunMethod(MethodCall call) throws Exception {
        int index = Objects.requireNonNull(call.argument("index"));
        String methodName = call.argument("methodName");
        Module module = modules.get(index);

        Object output;
        if (call.arguments instanceof HashMap<?, ?>) {
            IValue input = parseIValue((HashMap<?, ?>) call.arguments);
            output = module.runMethod(methodName, input);

            return output;
        } else {
            throw new Exception("Unknown type found for call.arguments: " + call.arguments.getClass());
        }

    }

    /**
     * Loads the model from the given asset path (flutter)
     *
     * @param args the arguments passed from the Flutter end, must contain "assetPath"
     * @return (int) the index of the model stored in the backend, on all the future calls using
     *  the `Model` class in Flutter end, it will be required to use the index
     */
    private int loadModelFromAsset(HashMap args) {
        String modelPath = pluginBinding.getFlutterAssets().getAssetFilePathBySubpath(Objects.requireNonNull(args.get("assetPath")).toString());
        Module module = Module.load(modelPath);
        modules.add(module);

        // return the index of this module
        return modules.size() - 1;
    }

    /**
     * Loads the model from the given file path
     *
     * @param args the arguments passed from flutter end, must contain "filePath"
     * @return (int) the index of the model stored in the backend, on all the future calls using
     *  the `Model` class in Flutter end, it will be required to use the index
     */
    private int loadModelFromFile(HashMap args) {
        String modelPath = Objects.requireNonNull(args.get("filePath")).toString();
        Module module = Module.load(modelPath);
        modules.add(module);

        // return the index of this module
        return modules.size() - 1;
    }

    /**
     * Converts the given IValue from Flutter end to the IValue in Android Java
     *
     * @param args the call.arguments received from flutter
     * @return (IValue) a representation needed to call methods on the model
     * @throws Exception if any exception happens during parsing the arguments
     */
    private IValue parseIValue(HashMap<?, ?> args) throws Exception {
        Object dataArg = Objects.requireNonNull(args.get("data"));
        String toDataType = Objects.requireNonNull(args.get("dtype")).toString();
        String classType = Objects.requireNonNull(args.get("class_type")).toString();

        IValue ivalue;

        switch (classType) {
            case "Value":
                ivalue = iValueFromValue(dataArg, toDataType);
                break;
            case "TensorValue":
                long[] shape = (long[])args.get("shape");
                ivalue = iValueFromTensorValue(dataArg, shape, toDataType);
                break;
            case "ImageTensor":
                double[] mean = (double[]) args.get("mean");
                double[] std = (double[]) args.get("std");
                ivalue = iValueFromImageTensor(dataArg, mean, std);
                break;
            default:
                throw new Exception("Unknown classType: " + classType + " supported ones are: Value, TensorValue, ImageTensor");
        }

        return ivalue;
    }

    /**
     * Runs the 'forward' method of the model with the given arguments
     *
     * @param call the `MethodCall` received from flutter end
     * @return (Object) the results of the forward function on the model
     * @throws Exception if any error happens during parsing the input or during forward call
     */
    private Object modelForward(MethodCall call) throws Exception {

        int index = Objects.requireNonNull(call.argument("index"));
        Module module = modules.get(index);

        Object output;
        if (call.arguments instanceof HashMap<?, ?>) {
            IValue input = parseIValue((HashMap<?, ?>) call.arguments);
            output = module.forward(input);

            return output;
        } else {
            throw new Exception("Unknown type found for call.arguments: " + call.arguments.getClass());
        }


    }

    /**
     * calls the 'forward' function on the Model, but with variable list of inputs
     *
     * @param call the `MethodCall` received from the flutter end
     * @return (Object) the output from the model forward function
     * @throws Exception if any error happens during parsing of arguments or during the forward
     *  function call
     */
    private Object modelForwardList(MethodCall call) throws Exception {
        int index = Objects.requireNonNull(call.argument("index"));
        Module module = modules.get(index);

        ArrayList<HashMap<?, ?>> inputsArg = call.argument("inputs");

        assert inputsArg != null;
        assert module != null;

        ArrayList<IValue> inputs = new ArrayList<>();
        // prepare the input arguments
        for (HashMap<?, ?> arg : inputsArg) {
            inputs.add(parseIValue(arg));
        }

        // convert the ArrayList to IValue[] that can be sent to the forward function with variable
        // list of arguments
        Object output = module.forward(inputs.toArray(new IValue[0]));

        return output;
    }

    /**
     * Parses the IValue from a Value type received from flutter
     *  IValue
     *
     * @param data the data received from flutter
     * @param toType the datatype to convert to, can only be BOOL, INT, FLOAT, STR
     * @return (IValue) the IValue representation of the data
     * @throws Exception throws an Exception if any error occurs during conversion
     */
    @NotNull
    private IValue iValueFromValue(Object data, String toType) throws Exception {
        @NotNull IValue result;
        switch (toType) {
            case "BOOL": {
                boolean _data = (Boolean) data;
                result = IValue.from(_data);
                break;
            }
            case "INT": {
                long _data;
                if (data instanceof Integer) {
                    _data = ((Integer) data).longValue();
                } else if (data instanceof Long) {
                    _data = ((Long) data);
                } else {
                    throw new Exception("Invalid data type received in Value: " + data.getClass());
                }
                result = IValue.from(_data);
                break;
            }
            case "FLOAT": {
                double _data = (Double) data;
                result = IValue.from(_data);
                break;
            }
            case "STR": {
                String _data = (String) data;
                result = IValue.from(_data);
                break;
            }
            default:
                throw new Exception("Invalid toType: " + toType);
        }
        return result;
    }

    /**
     * Creates an IValue from the given data Tensor, the allowed conversion to Tensor supported are
     *  UINT8, INT8, INT32, FLOAT32, INT64, FLOAT64
     *
     *  The data is first converted to the Tensor type defined, and then this Tensor is converted to
     *      IValue, since the forward function in Module expects IValue(s) only
     *
     * @param data The data received from flutter
     * @param shape The shape to resize to when converting to a Tensor
     * @param toType The Tensor type to convert to
     * @return (IValue) The Tensor converted to IValue
     * @throws Exception throws Exception if any error happens during the conversion process
     */
    @NotNull
    private IValue iValueFromTensorValue(Object data, long[] shape, String toType) throws Exception {
        @NotNull IValue result;
        switch (toType) {
            case "UINT8":
            case "INT8": {
                byte[] _data;
                if (data instanceof byte[]) {
                    byte[] temp = (byte[]) data;
                    _data = Bytes.toArray(Bytes.asList(temp));
                } else if (data instanceof int[]) {
                    int[] temp = (int[]) data;
                    _data = Bytes.toArray(Ints.asList(temp));
                } else if (data instanceof long[]) {
                    long[] temp = (long[]) data;
                    _data = Bytes.toArray(Longs.asList(temp));
                } else if (data instanceof double[]) {
                    double[] temp = (double[]) data;
                    _data = Bytes.toArray(Doubles.asList(temp));
                } else {
                    throw new Exception("Invalid data type for conversion to " + toType + " : " + data.getClass());
                }

                if (toType.equals("UINT8")) {
                    Tensor tensor = Tensor.fromBlobUnsigned(_data, shape);
                    result = IValue.from(tensor);
                    break;
                } else {
                    Tensor tensor = Tensor.fromBlob(_data, shape);
                    result = IValue.from(tensor);
                    break;
                }
            }
            case "FLOAT32": {
                float[] _data;
                if (data instanceof byte[]) {
                    byte[] temp = (byte[]) data;
                    _data = Floats.toArray(Bytes.asList(temp));
                } else if (data instanceof int[]) {
                    int[] temp = (int[]) data;
                    _data = Floats.toArray(Ints.asList(temp));
                } else if (data instanceof long[]) {
                    long[] temp = (long[]) data;
                    _data = Floats.toArray(Longs.asList(temp));
                } else if (data instanceof double[]) {
                    double[] temp = (double[]) data;
                    _data = Floats.toArray(Doubles.asList(temp));
                } else {
                    throw new Exception("Invalid data type for conversion to " + toType + " : " + data.getClass());
                }
                Tensor tensor = Tensor.fromBlob(_data, shape);
                result = IValue.from(tensor);
                break;
            }
            case "INT32": {
                int[] _data;
                if (data instanceof byte[]) {
                    byte[] temp = (byte[]) data;
                    _data = Ints.toArray(Bytes.asList(temp));
                } else if (data instanceof int[]) {
                    int[] temp = (int[]) data;
                    _data = Ints.toArray(Ints.asList(temp));
                } else if (data instanceof long[]) {
                    long[] temp = (long[]) data;
                    _data = Ints.toArray(Longs.asList(temp));
                } else if (data instanceof double[]) {
                    double[] temp = (double[]) data;
                    _data = Ints.toArray(Doubles.asList(temp));
                } else {
                    throw new Exception("Invalid data type for conversion to " + toType + " : " + data.getClass());
                }
                Tensor tensor = Tensor.fromBlob(_data, shape);
                result = IValue.from(tensor);
                break;
            }
            case "FLOAT64": {
                double[] _data;
                if (data instanceof byte[]) {
                    byte[] temp = (byte[]) data;
                    _data = Doubles.toArray(Bytes.asList(temp));
                } else if (data instanceof int[]) {
                    int[] temp = (int[]) data;
                    _data = Doubles.toArray(Ints.asList(temp));
                } else if (data instanceof long[]) {
                    long[] temp = (long[]) data;
                    _data = Doubles.toArray(Longs.asList(temp));
                } else if (data instanceof double[]) {
                    double[] temp = (double[]) data;
                    _data = Doubles.toArray(Doubles.asList(temp));
                } else {
                    throw new Exception("Invalid data type for conversion to " + toType + " : " + data.getClass());
                }
                Tensor tensor = Tensor.fromBlob(shape, _data);
                result = IValue.from(tensor);
                break;
            }
            case "INT64": {
                long[] _data;
                if (data instanceof byte[]) {
                    byte[] temp = (byte[]) data;
                    _data = Longs.toArray(Bytes.asList(temp));
                } else if (data instanceof int[]) {
                    int[] temp = (int[]) data;
                    _data = Longs.toArray(Ints.asList(temp));
                } else if (data instanceof long[]) {
                    long[] temp = (long[]) data;
                    _data = Longs.toArray(Longs.asList(temp));
                } else if (data instanceof double[]) {
                    double[] temp = (double[]) data;
                    _data = Longs.toArray(Doubles.asList(temp));
                } else {
                    throw new Exception("Invalid data type for conversion to " + toType + " : " + data.getClass());
                }
                Tensor tensor = Tensor.fromBlob(_data, shape);
                result = IValue.from(tensor);
                break;
            }
            default:
                throw new Exception("Invalid toType : " + toType);
        }
        return result;
    }

    /**
     * Creates an IValue from a given Image data
     *
     * The Image data received from flutter is of byte[], which is then converted to a Bitmap and then to a
     * Float32Tensor, and also applying the mean and std for the image, then an IValue is created from this Tensor
     * Internally we are using TensorImageUtils.bitmapToFloat32Tensor
     *
     * @param data The image data, only allowed type is byte[]
     * @param mean The mean of the image
     * @param std The standard deviation of the image
     * @return (IValue) an IValue representation of the Image Tensor
     * @throws Exception throws an Exception if happens during the conversion process
     */
    @NotNull
    private IValue iValueFromImageTensor(Object data, double[] mean, double[] std) throws Exception {
        @NotNull IValue result;
        float[] mean_ = Floats.toArray(Doubles.asList(mean));
        float[] std_ = Floats.toArray(Doubles.asList(std));
        if (data instanceof byte[]) {
            byte[] _data = (byte[]) data;
            Bitmap imageBitmap = BitmapFactory.decodeByteArray(_data, 0, _data.length);
            Tensor imageTensor = TensorImageUtils.bitmapToFloat32Tensor(imageBitmap, mean_, std_);
            result = IValue.from(imageTensor);
        } else {
            throw new Exception("Invalid type received for ImageTensor : " + data.getClass());
        }
        return result;
    }

    @Override
    public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
        channel.setMethodCallHandler(null);
    }
}

The point of this library is to provide a very flexible interface for the flutter
users to use the PyTorch library.

PyTorch uses jit::traced modules, which are basically C++ modules that can be then
used in anywhere which has the jit compiler, in Flutter we are using the pytorch_android
backend that does so.

So jit::traced modules are not only for ML models, they can also be used for normal functions
and classes, so to support this we have module.runMethod that can run an arbitrary method from
the module given the function name and module.forward which will run the forward function of
the module.

The problem is that we need to support a wide range of values, It's not that the modules will
always take in a Tensor and spit out a Tensor, they can also take in normal bool, int, float
kind of values, and to support all of this, it becomes a huge mess, so this library aims to
provide support of all kinds of values. Currently i am focusing only on getting this implemented
for Android, but a similar support can be given to iOS as well.

The point is that flutter supports these kind of data conversions: https://flutter.dev/docs/development/platform-integration/platform-channels?tab=android-channel-java-tab

I believe for the most part ML models are going to be used to Images and NLP.

Images in Dart are stored as Uint8List, so that shouldn't be a problem, as we can convert it to 
tensor of type INT32 or INT8 for anything.

torchvision_android has tools for converting an image to Tensor, so what we can do is pass in the
Image as Uint8List from Flutter to android, which will get converted to byte[] in Java, then use
BitmapFactory.decodeByteArray to convert this byte[] to a Bitmap, now it can be converted to a tensor

So finally here is my designed Interface

on Flutter user can send in values:
    - Uint8List
        - dtype.IMAGE_TENSOR
        - shape is required, for images this will only be `[H X W]` otherwise [`N X C X H X W`]
        
New thought process:

I can create different classes for different datatypes and their conversions that are supported in
them, for example, `ImageTensor` can be one class, which can hold only Uint8List data, and also the image shape
along with normalization values if required, these will be
subclasses of `ValueBackend` (abstract class), `TensorValue` can be one class which will only hold lists of type Uint8List,
Int32List, Int64List, Float64List, and these will then be converted to defined torch.type,
IValue will only support bool, int, double, and String, these will strictly be non-Tensors, so they
wont have shape property.

class Value will be used for non-Tensor type

ToMap function in IValue will be overloaded by TensorValue and ImageValue as required. on the Android
Backend we will check the class type as well, if it was Value, TensorValue or ImageTensor, based on that
and the values stored in them, we will do the appropriate conversions to the Android `IValue` type

Guava APIs look good: https://github.com/google/guava/wiki/PrimitivesExplained

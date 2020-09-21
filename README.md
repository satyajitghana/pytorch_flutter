# pytorch_flutter

A package to run PyTorch models on flutter

>What is different in this package ?
>
>It uses the latest `pytorch_android:1.6.0` and `pytorch_android_torchvision:1.6.0` on Android. I still have to work on the iOS implementation.

## Example

```dart
import 'package:pytorch_flutter/model.dart';
import 'package:pytorch_flutter/pytorch_flutter.dart';

class PlateDetector extends StatefulWidget {
  PlateDetector({Key key}) : super(key: key);

  @override
  _PlateDetectorState createState() => _PlateDetectorState();
}

class _PlateDetectorState extends State<PlateDetector> {

  Model yolov5s;

  void initModel() async {
    Random random = Random();
    yolov5s = await PyTorchFlutter.loadModel('assets/yolov5s.torchscript.pt');
    // warm up the model
    var input = List<double>.generate(3 * 640 * 640, (_) => random.nextInt(255) / 255.0);

    List predictions = await yolov5s.getPredictionList(input, [1, 3, 640, 640], DType.float32, DType.float32);

    print(predictions);
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        child: RaisedButton(
          onPressed: () => initModel(),
          child: Text('Run !'),
        ),
      ),
    );
  }

  @override
  void initState() {
    super.initState();
  }
}
```

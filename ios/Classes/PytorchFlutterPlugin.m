#import "PytorchFlutterPlugin.h"
#if __has_include(<pytorch_flutter/pytorch_flutter-Swift.h>)
#import <pytorch_flutter/pytorch_flutter-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "pytorch_flutter-Swift.h"
#endif

@implementation PytorchFlutterPlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftPytorchFlutterPlugin registerWithRegistrar:registrar];
}
@end

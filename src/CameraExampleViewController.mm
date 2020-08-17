// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#import "MySecondView.h"
#import "CameraExampleViewController.h"
#import <AssertMacros.h>
#import <AssetsLibrary/AssetsLibrary.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>

#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <queue>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/string_util.h"
#if TFLITE_USE_GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif

#define LOG(x) std::cerr

namespace {

// If you have your own model, modify this to the file name, and make sure
// you've added the file to your app resources too.
#if TFLITE_USE_GPU_DELEGATE
// GPU Delegate only supports float model now.
//NSString* model_file_name = @"mobilenet_v1_1.0_224";
#else
//NSString* model_file_name = @"mobilenet_quant_v1_224";
NSString* model_file_name = @"EE-ContiPhoto-version1";
#endif
NSString* model_file_type = @"tflite";
// If you have your own model, point this to the labels file.
//NSString* labels_file_name = @"labels";
NSString* labels_file_name = @"6label";
NSString* labels_file_type = @"txt";


// These dimensions need to match those the model was trained with.
const int wanted_input_width = 224;
const int wanted_input_height = 448;
const int wanted_input_channels = 3;
const float input_mean = 0;
const float input_std = 1;
const std::string input_layer_name = "input";
const std::string output_layer_name = "softmax1";

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
  NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (file_path == NULL) {
    LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "." << [extension UTF8String]
               << "' in bundle.";
  }
  return file_path;
}

void LoadLabels(NSString* file_name, NSString* file_type, std::vector<std::string>* label_strings) {
  NSString* labels_path = FilePathForResourceName(file_name, file_type);
  if (!labels_path) {
    LOG(ERROR) << "Failed to find model proto at" << [file_name UTF8String]
               << [file_type UTF8String];
  }
  std::ifstream t;
  t.open([labels_path UTF8String]);
  std::string line;
  while (t) {
    std::getline(t, line);
    label_strings->push_back(line);
  }
  t.close();
}

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
void GetTopN(
    const float* prediction, const int prediction_size, const int num_results,
    const float threshold, std::vector<std::pair<float, int> >* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
                      std::greater<std::pair<float, int> > >
      top_result_pq;

  const long count = prediction_size;
  for (int i = 0; i < count; ++i) {
    const float value = prediction[i];
    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

// Preprocess the input image and feed the TFLite interpreter buffer for a float model.
void ProcessInputWithFloatModel(
    uint8_t* input, float* buffer, int image_width, int image_height, int image_channels) {
  for (int y = 0; y < wanted_input_height; ++y) {
    float* out_row = buffer + (y * wanted_input_width * wanted_input_channels);
    for (int x = 0; x < wanted_input_width; ++x) {
      const int in_x = (y * image_width) / wanted_input_width;
      const int in_y = (x * image_height) / wanted_input_height;
      uint8_t* input_pixel =
          input + (in_y * image_width * image_channels) + (in_x * image_channels);
      float* out_pixel = out_row + (x * wanted_input_channels);
      for (int c = 0; c < wanted_input_channels; ++c) {
          out_pixel[c] = (input_pixel[c] - input_mean) / input_std;
      }
    }
  }
}


/* color change */
void ProviderReleaseData (void *info, const void *data, size_t size){
        free((void*)data);
    }

}  // namespace

@interface CameraExampleViewController (InternalMethods)
- (void)setupAVCapture;
- (void)teardownAVCapture;
- (void)test;
- (UIImage*)imageBlackToTransparent;

@end

@implementation CameraExampleViewController {
  std::unique_ptr<tflite::FlatBufferModel> model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  TfLiteDelegate* delegate;
}


//button setup
- (IBAction)analyze:(id)sender {
    //testing
    if(imageView_1.image && imageView_2.image){
        [self performSegueWithIdentifier:@"secondView" sender:self];
    }
}

-(void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender{
    if(~(imageView_1.image && imageView_2.image)){
        MySecondView *controller =(MySecondView *)segue.destinationViewController;
        controller.content_1 =@"pictures aren't enough";
    }
    if(imageView_1.image && imageView_2.image){
        MySecondView *controller =(MySecondView *)segue.destinationViewController;
        UIImage* merge=[self mergeImage];
        CVPixelBufferRef pixelBuffer = [self imageToRGBPixelBuffer:[self normalize:merge]];
        CFRetain(pixelBuffer);
        [self runModelOnFrame:pixelBuffer];
        CFRelease(pixelBuffer);
        
        NSString* Text_1 = [[NSString alloc] initWithFormat:@"%d", firstlabel];
        NSLog(@"firstlabel=%d",firstlabel);
        controller.content_1 =Text_1;
        
        NSString* Text_2= [[NSString alloc] initWithFormat:@"%d", secondlabel];
        NSLog(@"firstlabel=%d",secondlabel);
        controller.content_2 =Text_2;

    }
}

- (AVCaptureDevice *)backCamera {
    NSArray *devices;
    if (@available(iOS 10.0, *)) {
        if (@available(iOS 11.1, *)) {
            AVCaptureDeviceDiscoverySession *session = [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInTrueDepthCamera, AVCaptureDeviceTypeBuiltInWideAngleCamera] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionBack];
            devices = session.devices;
        } else {
            // Fallback on earlier versions
        }
    }else{
        devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    }
    for (AVCaptureDevice *device in devices) {
        if ([device position] == AVCaptureDevicePositionBack) {
            return device;
        }
    }
    return nil;
}



AVCaptureStillImageOutput *StillImageOutput;
- (void)setupAVCapture{
    AVCaptureSession *session = [[AVCaptureSession alloc]init];
    session.sessionPreset = AVCaptureSessionPresetPhoto;
    //Add device
    AVCaptureDevice *device =
    [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    //Input
    AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:nil];
    if (!input){NSLog(@"No Input");}
    [session addInput:input];
    //Output
    AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
    [session addOutput:output];
    output.videoSettings =
    @{ (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA) };
    //Preview Layer
    AVCaptureVideoPreviewLayer *previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:session];
    UIView *myView = previewView;
    previewLayer.frame = myView.bounds;
    previewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
    [previewView.layer addSublayer:previewLayer];
    //Start capture session
    [session startRunning];
    StillImageOutput =[[AVCaptureStillImageOutput alloc] init];
    NSDictionary *outputSettings =[[NSDictionary alloc] initWithObjectsAndKeys:AVVideoCodecTypeJPEG, AVVideoCodecKey, nil];
    [StillImageOutput setOutputSettings:outputSettings];
    [session addOutput:StillImageOutput];
    //Start capture session
    [session startRunning];
}


- (CVPixelBufferRef)imageToRGBPixelBuffer:(UIImage *)image {
    CGSize frameSize = CGSizeMake(CGImageGetWidth(image.CGImage),CGImageGetHeight(image.CGImage));
    NSDictionary *options =
    [NSDictionary dictionaryWithObjectsAndKeys:[NSNumber numberWithBool:YES],kCVPixelBufferCGImageCompatibilityKey,[NSNumber numberWithBool:YES],kCVPixelBufferCGBitmapContextCompatibilityKey,nil];
    CVPixelBufferRef pxbuffer = NULL;
    CVReturn status =
    CVPixelBufferCreate(kCFAllocatorDefault, frameSize.width, frameSize.height,kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)options, &pxbuffer);
    NSParameterAssert(status == kCVReturnSuccess && pxbuffer != NULL);
    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);
    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(pxdata, frameSize.width, frameSize.height,8, CVPixelBufferGetBytesPerRow(pxbuffer),rgbColorSpace,(CGBitmapInfo)kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    CGContextDrawImage(context, CGRectMake(0, 0, CGImageGetWidth(image.CGImage),CGImageGetHeight(image.CGImage)), image.CGImage);
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);
    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);
    return pxbuffer;
}

- (void)teardownAVCapture {
  [previewLayer removeFromSuperlayer];
}


- (UIImage *)image:(UIImage *)image rotation:(UIImageOrientation)orientation
{
    long double rotate = 0.0;
    CGRect rect;
    float translateX = 0;
    float translateY = 0;
    float scaleX = 1.0;
    float scaleY = 1.0;
    
    switch (orientation) {
        case UIImageOrientationLeft:
            rotate = M_PI_2;
            rect = CGRectMake(0, 0, image.size.height, image.size.width);
            translateX = 0;
            translateY = -rect.size.width;
            scaleY = rect.size.width/rect.size.height;
            scaleX = rect.size.height/rect.size.width;
            break;
        case UIImageOrientationRight:
            rotate = 3 * M_PI_2;
            rect = CGRectMake(0, 0, image.size.height, image.size.width);
            translateX = -rect.size.height;
            translateY = 0;
            scaleY = rect.size.width/rect.size.height;
            scaleX = rect.size.height/rect.size.width;
            break;
        case UIImageOrientationDown:
            rotate = M_PI;
            rect = CGRectMake(0, 0, image.size.width, image.size.height);
            translateX = -rect.size.width;
            translateY = -rect.size.height;
            break;
        default:
            rotate = 0.0;
            rect = CGRectMake(0, 0, image.size.width, image.size.height);
            translateX = 0;
            translateY = 0;
            break;
    }
    
    UIGraphicsBeginImageContext(rect.size);
    CGContextRef context = UIGraphicsGetCurrentContext();
    //做CTM变换
    CGContextTranslateCTM(context, 0.0, rect.size.height);
    CGContextScaleCTM(context, 1.0, -1.0);
    CGContextRotateCTM(context, rotate);
    CGContextTranslateCTM(context, translateX, translateY);
    
    CGContextScaleCTM(context, scaleX, scaleY);
    //绘制图片
    CGContextDrawImage(context, CGRectMake(0, 0, rect.size.width, rect.size.height), image.CGImage);
    
    UIImage *newPic = UIGraphicsGetImageFromCurrentImageContext();
    
    return newPic;
}

- (UIImage *)reSizeImage:(UIImage *)image toSize:(CGSize)reSize{
    UIGraphicsBeginImageContext(CGSizeMake(reSize.width, reSize.height));
    [image drawInRect:CGRectMake(0, 0, reSize.width, reSize.height)];
    UIImage *reSizeImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return reSizeImage;
}

- (UIImage*)normalize:(UIImage*)image{
    const int imageWidth = image.size.width;
    const int imageHeight = image.size.height;
    size_t      bytesPerRow = imageWidth * 4;
    uint32_t* rgbImageBuf = (uint32_t*)malloc(bytesPerRow * imageHeight);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(rgbImageBuf, imageWidth, imageHeight, 8, bytesPerRow, colorSpace,
                                                 kCGBitmapByteOrder32Little | kCGImageAlphaNoneSkipLast);
    CGContextDrawImage(context, CGRectMake(0, 0, imageWidth, imageHeight), image.CGImage);
    // visit every pixel
    int pixelNum = imageWidth * imageHeight;
    uint32_t* pCurPtr = rgbImageBuf;
    
    
    for (int i = 0; i < pixelNum; i++, pCurPtr++){
        uint8_t* ptr = (uint8_t*) pCurPtr;
        ptr[1] = [self ptrIs: ptr[1] meanIs:103.94 scaleIs:0.017];
        ptr[2] = [self ptrIs: ptr[2] meanIs:116.78 scaleIs:0.017];
        ptr[3] = [self ptrIs: ptr[3] meanIs:123.68 scaleIs:0.017];
    }
    
    // output the image
    CGDataProviderRef dataProvider = CGDataProviderCreateWithData(NULL, rgbImageBuf, bytesPerRow * imageHeight, ProviderReleaseData);
    CGImageRef imageRef = CGImageCreate(imageWidth, imageHeight, 8, 32, bytesPerRow, colorSpace,
                                        kCGImageAlphaLast | kCGBitmapByteOrder32Little, dataProvider,
                                        NULL, true, kCGRenderingIntentDefault);
    CGDataProviderRelease(dataProvider);
    UIImage* resultUIImage = [UIImage imageWithCGImage:imageRef];
    
    // release the space
    CGImageRelease(imageRef);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    return resultUIImage;
}

- (float) ptrIs: (float)ptr meanIs:(float)mean scaleIs:(float)standard{
    ptr =(ptr-mean)* scale;
    if(ptr<=0.5) return 0;
    else if(0.5<ptr&&ptr<1.5) return 1;
    else if(1.5<=ptr&&ptr<=2.5) return 2;
    else return 3;
}

- (UIImage*)pointCloud:(UIImage*)image{
    const int imageWidth = image.size.width;
    const int imageHeight = image.size.height;
    size_t      bytesPerRow = imageWidth * 4;
    uint32_t* rgbImageBuf = (uint32_t*)malloc(bytesPerRow * imageHeight);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(rgbImageBuf, imageWidth, imageHeight, 8, bytesPerRow, colorSpace,
                                                 kCGBitmapByteOrder32Little | kCGImageAlphaNoneSkipLast);
    CGContextDrawImage(context, CGRectMake(0, 0, imageWidth, imageHeight), image.CGImage);
    // get the number of the pixel
    int pixelNum = imageWidth * imageHeight;
    uint32_t* pCurPtr = rgbImageBuf;
    int count=0;
    for (int i = 0; i < pixelNum; i++, pCurPtr++){
        count++;
    }
    //initial
    pCurPtr = rgbImageBuf;
    float rate=0.3;
    int counter=0;
    for (int i = 0; i < pixelNum; i++, pCurPtr++){
        uint8_t* ptr=(uint8_t*)pCurPtr;
        // uppper part clip by rate%
        if(counter<=rate*count){
            [self dropPixelRate:0.3 ptrIs: ptr];
        }
        else if(counter>rate*count&&counter<(1-rate)*count){
            [self dropPixelRate:0.2 ptrIs: ptr];
        }
        else{
            [self dropPixelRate:0.3 ptrIs: ptr];
        }
        counter++;
    }
    
    // output the picture
    CGDataProviderRef dataProvider = CGDataProviderCreateWithData(NULL, rgbImageBuf, bytesPerRow * imageHeight, ProviderReleaseData);
    CGImageRef imageRef = CGImageCreate(imageWidth, imageHeight, 8, 32, bytesPerRow, colorSpace,
                                        kCGImageAlphaLast | kCGBitmapByteOrder32Little, dataProvider,
                                        NULL, true, kCGRenderingIntentDefault);
    CGDataProviderRelease(dataProvider);
    UIImage* resultUIImage = [UIImage imageWithCGImage:imageRef];
    
    // release the space
    CGImageRelease(imageRef);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    return resultUIImage;
}

-(void) dropPixelRate: (float)dropRate ptrIs:(uint8_t*) ptr{
    int value = (arc4random() % 100) + 1;//value range from 1 to 100
    if(value<(dropRate * 100)){
        for(int j=0;j<1;j++){
            uint8_t* ptr_3= ptr;
            ptr_3[1] = 0;
            ptr_3[2] = 0;
            ptr_3[3] = 0;
            ptr++;
        }
    }
}

- (AVCaptureVideoOrientation)avOrientationForDeviceOrientation:
    (UIDeviceOrientation)deviceOrientation {
  AVCaptureVideoOrientation result = (AVCaptureVideoOrientation)(deviceOrientation);
  if (deviceOrientation == UIDeviceOrientationLandscapeLeft)
    result = AVCaptureVideoOrientationLandscapeRight;
  else if (deviceOrientation == UIDeviceOrientationLandscapeRight)
    result = AVCaptureVideoOrientationLandscapeLeft;
  return result;
}

UIImage *image_1;
-(IBAction)takephoto_1:(id)sender{
    AVCaptureConnection *videoConnection=nil;
    for (AVCaptureConnection *connection in StillImageOutput.connections) {
        for (AVCaptureInputPort *port in [connection inputPorts]){
            if([[port mediaType] isEqual:AVMediaTypeVideo]){
                videoConnection =connection;
                break;
            }
            
        }
    }
    //show in the UIimage
    [StillImageOutput captureStillImageAsynchronouslyFromConnection:videoConnection completionHandler:^(CMSampleBufferRef imageDataSampleBuffer, NSError *error){
        if(imageDataSampleBuffer !=NULL){
            NSData *imageData =[AVCaptureStillImageOutput jpegStillImageNSDataRepresentation:imageDataSampleBuffer];
            UIImage *image =[UIImage imageWithData:imageData];
            CGSize size = CGSizeMake(224, 224);
            image_1=[self reSizeImage:[self image:image rotation:UIImageOrientationRight] toSize:size];
            image_1=[self pointCloud:image_1];
            self->imageView_1.image =image_1;
        }
    }];
}

UIImage *image_2;
- (IBAction)takephoto_2:(id)sender {
    AVCaptureConnection *videoConnection=nil;
    for (AVCaptureConnection *connection in StillImageOutput.connections) {
        for (AVCaptureInputPort *port in [connection inputPorts]){
            if([[port mediaType] isEqual:AVMediaTypeVideo]){
                videoConnection =connection;
                break;
            }
            
        }
    }
    
    [StillImageOutput captureStillImageAsynchronouslyFromConnection:videoConnection completionHandler:^(CMSampleBufferRef imageDataSampleBuffer, NSError *error){
        if(imageDataSampleBuffer !=NULL){
            NSData *imageData =[AVCaptureStillImageOutput jpegStillImageNSDataRepresentation:imageDataSampleBuffer];
            UIImage *image =[UIImage imageWithData:imageData];
            //show in the UIimage
            CGSize size=CGSizeMake(224, 224);
            image_2=[self reSizeImage:[self image:image rotation:UIImageOrientationRight] toSize:size];
            image_2=[self pointCloud:image_2];
            self->imageView_2.image =image_2;
            
            //做testing
            CVPixelBufferRef pixelBuffer = [self imageToRGBPixelBuffer:[self normalize:image_2]];
            CFRetain(pixelBuffer);
            [self runModelOnFrame:pixelBuffer];
            CFRelease(pixelBuffer);
        }
    }];
}

- (UIImage*)mergeImage{
    // get size of the first image
    CGImageRef firstImageRef = image_1.CGImage;
    CGFloat firstWidth = CGImageGetWidth(firstImageRef);
    CGFloat firstHeight = CGImageGetHeight(firstImageRef);
    
    // get size of the second image
    CGImageRef secondImageRef = image_2.CGImage;
    CGFloat secondWidth = CGImageGetWidth(secondImageRef);
    CGFloat secondHeight = CGImageGetHeight(secondImageRef);
    
    // build merged size
    CGSize mergedSize = CGSizeMake(firstWidth,(firstHeight+secondHeight));
    // capture image context ref
    UIGraphicsBeginImageContext(mergedSize);
    
    //Draw images onto the context
    [image_1 drawInRect:CGRectMake(0, 0, firstWidth, firstHeight)];
    //[second drawInRect:CGRectMake(firstWidth, 0, secondWidth, secondHeight)];
    [image_2 drawInRect:CGRectMake(0, firstHeight+1, secondWidth, secondHeight) blendMode:kCGBlendModeNormal alpha:1.0];
    
    // assign context to new UIImage
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    // end context
    UIGraphicsEndImageContext();
    
    return newImage;
}


- (void)runModelOnFrame:(CVPixelBufferRef)pixelBuffer {
  assert(pixelBuffer != NULL);

  OSType sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
  assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
         sourcePixelFormat == kCVPixelFormatType_32BGRA);

  const int sourceRowBytes = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);
  const int image_width = (int)CVPixelBufferGetWidth(pixelBuffer);
  const int fullHeight = (int)CVPixelBufferGetHeight(pixelBuffer);

  CVPixelBufferLockFlags unlockFlags = kNilOptions;
  CVPixelBufferLockBaseAddress(pixelBuffer, unlockFlags);

  unsigned char* sourceBaseAddr = (unsigned char*)(CVPixelBufferGetBaseAddress(pixelBuffer));
  int image_height;
  unsigned char* sourceStartAddr;
  
  image_height = fullHeight;
  sourceStartAddr = sourceBaseAddr;
  
  const int image_channels = 4;
  assert(image_channels >= wanted_input_channels);
  uint8_t* in = sourceStartAddr;

  int input = interpreter->inputs()[0];
  TfLiteTensor *input_tensor = interpreter->tensor(input);

  bool is_quantized;
  switch (input_tensor->type) {
  case kTfLiteFloat32:
    is_quantized = false;
    break;
  case kTfLiteUInt8:
    is_quantized = true;
    break;
  default:
    NSLog(@"Input data type is not supported by this demo app.");
    return;
  }
    
  float* out = interpreter->typed_tensor<float>(input);
  ProcessInputWithFloatModel(in, out, image_width, image_height, image_channels);
  //NSLog(@"%d %d %d", image_width, image_height, image_channels);
  if (interpreter->Invoke() != kTfLiteOk) {
    LOG(FATAL) << "Failed to invoke!";
  }
  
  // read output size from the output sensor
  const int output_tensor_index = interpreter->outputs()[0];
  TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_index);
  TfLiteIntArray* output_dims = output_tensor->dims;
  if (output_dims->size != 2 || output_dims->data[0] != 1) {
    LOG(FATAL) << "Output of the model is in invalid format.";
  }
  const int output_size = output_dims->data[1];

  const int kNumResults = 5;
  const float kThreshold = 0.1f;

  std::vector<std::pair<float, int> > top_results;

  float* output = interpreter->typed_output_tensor<float>(0);
  for (int i = 0; i < output_size; ++i) {
     NSLog (@"output[%d]=%f",i,output[i]);
  }
  GetTopN(output, output_size, kNumResults, kThreshold, &top_results);

  NSMutableDictionary* newValues = [NSMutableDictionary dictionary];
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    NSString* labelObject = [NSString stringWithUTF8String:labels[index].c_str()];
    NSNumber* valueObject = [NSNumber numberWithFloat:confidence];
    [newValues setObject:valueObject forKey:labelObject];
    //break;
  }
    
  [self setPredictionValues:newValues];
  CVPixelBufferUnlockBaseAddress(pixelBuffer, unlockFlags);
  CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

- (void)dealloc {
#if TFLITE_USE_GPU_DELEGATE
  if (delegate) {
    DeleteGpuDelegate(delegate);
  }
#endif
  [self teardownAVCapture];
}

- (void)didReceiveMemoryWarning {
  [super didReceiveMemoryWarning];
}

- (void)viewDidLoad {
  [super viewDidLoad];
  labelLayers = [[NSMutableArray alloc] init];
  oldPredictionValues = [[NSMutableDictionary alloc] init];

  NSString* graph_path = FilePathForResourceName(model_file_name, model_file_type);
  model = tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]);
  if (!model) {
    LOG(FATAL) << "Failed to mmap model " << graph_path;
  }
  LOG(INFO) << "Loaded model " << graph_path;
  model->error_reporter();
  LOG(INFO) << "resolved reporter";

  tflite::ops::builtin::BuiltinOpResolver resolver;
  LoadLabels(labels_file_name, labels_file_type, &labels);

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);

#if TFLITE_USE_GPU_DELEGATE
  GpuDelegateOptions options;
  options.allow_precision_loss = true;
  options.wait_type = GpuDelegateOptions::WaitType::kActive;
  delegate = NewGpuDelegate(&options);
  interpreter->ModifyGraphWithDelegate(delegate);
#endif

  // Explicitly resize the input tensor.
  {
    int input = interpreter->inputs()[0];
    std::vector<int> sizes = {1, 224, 448 , 3};
    interpreter->ResizeInputTensor(input, sizes);
  }
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter";
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  [self setupAVCapture];
}

- (void)viewDidUnload {
  [super viewDidUnload];
}

- (void)viewWillAppear:(BOOL)animated {
  [super viewWillAppear:animated];
}

- (void)viewDidAppear:(BOOL)animated {
  [super viewDidAppear:animated];
}

- (void)viewWillDisappear:(BOOL)animated {
  [super viewWillDisappear:animated];
}

- (void)viewDidDisappear:(BOOL)animated {
  [super viewDidDisappear:animated];
}

-(BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation {
  return (interfaceOrientation == UIInterfaceOrientationPortrait);
}

- (BOOL)prefersStatusBarHidden {
  return YES;
}


- (void)setPredictionValues:(NSDictionary*)newValues {
  const float decayValue = 0.75f;
  const float updateValue = 0.25f;
  const float minimumThreshold = 0.01f;

  NSMutableDictionary* decayedPredictionValues = [[NSMutableDictionary alloc] init];
  for (NSString* label in oldPredictionValues) {
    NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
    const float oldPredictionValue = [oldPredictionValueObject floatValue];
    const float decayedPredictionValue = (oldPredictionValue * decayValue);
    if (decayedPredictionValue > minimumThreshold) {
      NSNumber* decayedPredictionValueObject = [NSNumber numberWithFloat:decayedPredictionValue];
      [decayedPredictionValues setObject:decayedPredictionValueObject forKey:label];
    }
  }
  oldPredictionValues = decayedPredictionValues;

  for (NSString* label in newValues) {
    NSNumber* newPredictionValueObject = [newValues objectForKey:label];
    NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
    if (!oldPredictionValueObject) {
      oldPredictionValueObject = [NSNumber numberWithFloat:0.0f];
    }
    const float newPredictionValue = [newPredictionValueObject floatValue];
    const float oldPredictionValue = [oldPredictionValueObject floatValue];
    const float updatedPredictionValue = (oldPredictionValue + (newPredictionValue * updateValue));
    NSNumber* updatedPredictionValueObject = [NSNumber numberWithFloat:updatedPredictionValue];
    [oldPredictionValues setObject:updatedPredictionValueObject forKey:label];
  }
  NSArray* candidateLabels = [NSMutableArray array];
  for (NSString* label in oldPredictionValues) {
    NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
    const float oldPredictionValue = [oldPredictionValueObject floatValue];
    if (oldPredictionValue > 0.0) {
      NSDictionary* entry = @{@"label" : label, @"value" : oldPredictionValueObject};
      candidateLabels = [candidateLabels arrayByAddingObject:entry];
    }
  }
  NSSortDescriptor* sort = [NSSortDescriptor sortDescriptorWithKey:@"value" ascending:NO];
  NSArray* sortedLabels =
      [candidateLabels sortedArrayUsingDescriptors:[NSArray arrayWithObject:sort]];
    
  NSLog(@"%@",sortedLabels);
  
  firstlabel=[[[sortedLabels firstObject] objectForKey:@"label"] intValue];
  secondlabel=[[[sortedLabels objectAtIndex:1] objectForKey:@"label"] intValue];
 
}


- (void)removeAllLabelLayers {
  for (CATextLayer* layer in labelLayers) {
    [layer removeFromSuperlayer];
  }
  [labelLayers removeAllObjects];
}

- (void)addLabelLayerWithText:(NSString*)text
                      originX:(float)originX
                      originY:(float)originY
                        width:(float)width
                       height:(float)height
                    alignment:(NSString*)alignment {
  CFTypeRef font = (CFTypeRef) @"Menlo-Regular";
  const float fontSize = 12.0;
  const float marginSizeX = 5.0f;
  const float marginSizeY = 2.0f;

  const CGRect backgroundBounds = CGRectMake(originX, originY, width, height);
  const CGRect textBounds = CGRectMake((originX + marginSizeX), (originY + marginSizeY),
                                       (width - (marginSizeX * 2)), (height - (marginSizeY * 2)));

  CATextLayer* background = [CATextLayer layer];
  [background setBackgroundColor:[UIColor blackColor].CGColor];
  [background setOpacity:0.5f];
  [background setFrame:backgroundBounds];
  background.cornerRadius = 5.0f;

  [[self.view layer] addSublayer:background];
  [labelLayers addObject:background];

  CATextLayer* layer = [CATextLayer layer];
  [layer setForegroundColor:[UIColor whiteColor].CGColor];
  [layer setFrame:textBounds];
  [layer setAlignmentMode:alignment];
  [layer setWrapped:YES];
  [layer setFont:font];
  [layer setFontSize:fontSize];
  layer.contentsScale = [[UIScreen mainScreen] scale];
  [layer setString:text];

  [[self.view layer] addSublayer:layer];
  [labelLayers addObject:layer];
}

@end

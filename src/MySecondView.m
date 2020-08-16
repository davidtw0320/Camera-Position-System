//
//  MySecondView.m
//  tflite_camera_example
//
//  Created by 陳民健 on 2019/5/1.
//  Copyright © 2019 Google. All rights reserved.
//

#import "MySecondView.h"

@interface MySecondView ()

@end

@implementation MySecondView

- (void)viewDidLoad {
    [super viewDidLoad];
    
    // firstlabel
    int value_1 = [_content_1 intValue];
    switch (value_1) {
        case 1:
            _imageView1.image=[UIImage imageNamed:@"target.png"];
            break;
        case 2:
            _imageView2.image=[UIImage imageNamed:@"target.png"];
            break;
        case 3:
            _imageView3.image=[UIImage imageNamed:@"target.png"];
            break;
        case 4:
            _imageView4.image=[UIImage imageNamed:@"target.png"];
            break;
        case 5:
            _imageView5.image=[UIImage imageNamed:@"target.png"];
            break;
        case 6:
            _imageView6.image=[UIImage imageNamed:@"target.png"];
             break;
        default:
            break;
    }
    if([self.content_1 isEqual:@"pictures aren't enough"])self.label.text=self.content_1;
}

- (UIImage *)imageByApplyingAlpha:(CGFloat)alpha  image:(UIImage*)image
{
    UIGraphicsBeginImageContextWithOptions(image.size, NO, 0.0f);
    
    CGContextRef ctx = UIGraphicsGetCurrentContext();
    
    CGRect area = CGRectMake(0, 0, image.size.width, image.size.height);
    
    CGContextScaleCTM(ctx, 1, -1);
    
    CGContextTranslateCTM(ctx, 0, -area.size.height);
    
    CGContextSetBlendMode(ctx, kCGBlendModeMultiply);
    
    CGContextSetAlpha(ctx, alpha);
    
    CGContextDrawImage(ctx, area, image.CGImage);
    
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    
    UIGraphicsEndImageContext();
    
    return newImage;
}

- (IBAction)analyzeAgain:(id)sender{
        [self performSegueWithIdentifier:@"firstView" sender:self];
}

@end

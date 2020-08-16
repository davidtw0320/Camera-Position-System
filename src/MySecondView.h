//
//  MySecondView.h
//  tflite_camera_example
//
//  Created by 陳民健 on 2019/5/1.
//  Copyright © 2019 Google. All rights reserved.
//

#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface MySecondView : UIViewController

@property (strong, nonatomic) IBOutlet UILabel *label;
@property (strong, nonatomic) IBOutlet NSString *content_1;
@property (strong, nonatomic) IBOutlet NSString *content_2;
@property (strong, nonatomic) IBOutlet UIImageView *imageView1;
@property (strong, nonatomic) IBOutlet UIImageView *imageView2;
@property (strong, nonatomic) IBOutlet UIImageView *imageView3;
@property (strong, nonatomic) IBOutlet UIImageView *imageView4;
@property (strong, nonatomic) IBOutlet UIImageView *imageView5;
@property (strong, nonatomic) IBOutlet UIImageView *imageView6;

-(IBAction)analyzeAgain:(id)sender;
@end

NS_ASSUME_NONNULL_END

# **CPS (The Camera Position System)** 

## Description

CPS (The Camera Position System) applies trained B-CNN model to provide users with real-time service. Users need to take pictures as testing data. Then, CPS discards pixels randomly to simulate training point cloud data, normalize pixels to reduce the noise produced by random environments, and sent data into the Tensorflow interpreter. Finally, CPS predict the location according to testing set of pictures with user-friendly interface.

![image info](image/description.png)

***

## Prerequisites

+ Xcode Version 11.6

***

## Demo video

[![Watch the video](https://i.imgur.com/vRJmHuf.png?1)](https://drive.google.com/file/d/1LGRuJsA-jR51jpUwZw695J9G2o3ogRd4/view?usp=sharing)

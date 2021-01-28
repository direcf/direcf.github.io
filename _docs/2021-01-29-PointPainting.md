---
title: "PointPainting, DeepLabV3, SqueezeSegV2"
excerpt: "Sensor Fusion"
permalink: /docs/page1005/
author_profile: true
layout: single
classes: wide
comments: true
header:
    image: /assets/images/header2.jpg
---
## <span style="color:#3498DB">Assignment2.</span> 3D Semantic Segmentation

#### < Introduction >

Sensor fusion is one of important choices to develop 3d data analysis. Today, I introduce special fusion mehod, named PointPainting, using mono camera and lidar sensor. Concept is very simple.

<a href="https://imgur.com/ET3IX3J"><img src="https://i.imgur.com/ET3IX3J.png" title="source: imgur.com" /></a>

**Step1** Image semantic segmentation using DeepLabV3 or SqueezeSegV2.

**Step2** Project lidar data(point cloud) like image data. And concat image semantic segmentation to lidar data.

**Step3** Insert segmented point cloud data in 3d object detection network like PointRCNN or PointPillar.

It's just all process for sensor fusion. If think about it from input data view, it's just changed from 3d lidar data(x,y,z) to 3+(the number of segmentation class) input data size.

However, unfortunately it is hard to call it meaningful fusion technique. Because **<span style="color:#C7855C">image segmentation result just works on an assistant of 3d object detection.</span>** It's the reason why the author call it not 'parallel method', but 'sequential method'.

Additionally, 'seqential method' has a critical problem that is 'speed'. No matter how much 3d detection speed is high, if 2d image segmentation speed is low, it is useless. So the author suggests one solution 'pipelining' which means we use segmentation result of prior frame (consecutive).

<a href="https://imgur.com/g3BoY9m"><img src="https://i.imgur.com/g3BoY9m.png" title="source: imgur.com" /></a>


#### < Code Anaylsis >

**1. import module**


#### <DeepLabV3>


#### <SqueezeSegV2>
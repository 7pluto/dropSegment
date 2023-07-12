2023.7.11
通过seg切割叶片，并且提取mask区域，对mask区域做随机裁切，裁切的图像可以OTSU阈值处理，对于叶片只能用固定阈值处理，因为叶片周围还是有黑色背景，OTSU会将这部分算进去。

leafROI -- 阈值化处理 rangROI -- 分割叶片 随机裁切 阈值化处理

![Figure_1](https://github.com/7pluto/dropSegment/assets/50101024/8e7425a1-8cfc-4362-a315-1bd750c7cd5a)
![Figure_2](https://github.com/7pluto/dropSegment/assets/50101024/7a5dd156-561c-4a7b-ae93-6b49d4f0dab5)
![Figure_3](https://github.com/7pluto/dropSegment/assets/50101024/7537137a-9b3d-47d8-a434-8ee0f708eac7)
![Figure_4](https://github.com/7pluto/dropSegment/assets/50101024/1aaa56dc-2586-4f47-ac5f-4460d6eef233)
![leaf_90 0_20230711_195942](https://github.com/7pluto/dropSegment/assets/50101024/0b01ec43-e0e8-4680-99e7-5a34b540ecd3)

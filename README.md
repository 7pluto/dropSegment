2023.7.12  
通过seg切割叶片，并且提取mask区域，对mask区域做随机裁切，裁切的图像可以OTSU阈值处理，对于叶片只能用固定阈值处理，因为叶片周围还是有黑色背景，OTSU会将这部分算进去。

2023.7.14  
rangROI.py 通过fastsam提取叶片mask区域，对mask区域做随机裁切  
disposeROI.py 对图像做阈值化处理，增加通道分离的预处理，能够一定程度弥补二值化缺失的反射光区域。统计了轮廓的个数和面积  
K-means.py 对雾滴进行K-means聚类。  


![Figure_1](plt/Figure_1.png)
![Figure_2](plt/Figure_2.png)
![Figure_3](plt/Figure_3.png)
![Figure_4](plt/Figure_4.png)
![leaf_90 0_20230711_195942](plt/leaf_90.0_20230711_195942.png)

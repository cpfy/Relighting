

## Simple Blender

> 一些Blender的使用指北
>
> 2022-08-12更新



### 官方

* 官网（目前最新的3.2.2版本）：https://www.blender.org/

* 开源仓库，但仅为mirror存档状态：https://github.com/blender/blender

* 最重要的Blender手册：https://docs.blender.org/manual



### Screenshot

* 移除所有坐标轴等信息并拍照（貌似stack还有独立的Blender社区）

* 主要就是去除物体主视图右上角Gizmo与叠加层+全屏

* 说明：https://blender.stackexchange.com/questions/165532/turn-off-all-menus-for-a-screenshot



### 点云着色

BG 的 Ground Truth 打开之后只有单一黑色或橘色（选中）

（看来需要额外安装插件Add-ons）



#### Add-ons

导入方式：编辑-->偏好设置-->插件（或ctrl+alt+U 然鹅不好使）

本地安装直接选zip

* 导入方式：https://www.cgchan.com/static/doc/sceneskies/1.1/installation.html
* 此处选择 import-ply-as-verts 插件，Github地址为：https://github.com/TombstoneTumbleweedArt/import-ply-as-verts
* 原始的推荐与简单操作说明链接：https://blender.stackexchange.com/questions/257936/is-there-an-easy-way-to-import-point-clouds-with-colors-in-blender-3-2



#### Shader操作

上一链接中提到需要创建Shader with `Col` attribute

一篇挺好挺全的Blender Shader介绍教程：https://styly.cc/tips/blender_shader_beginner/

很简单的拖动，基于节点的工作流模式（node-based workflow）



* 快速新增节点快捷键： `Shift`+`A` 或点击 `添加` 
* 删除快捷键：`X`
* 貌似下面的geo 或所有面板都是此快捷键



#### Geometry Nodes

简略的入门教程：https://all3dp.com/2/blender-geometry-nodes-simply-explained/

很全、然而啥也看不懂的官方文档：https://docs.blender.org/manual/en/latest/modeling/geometry_nodes/index.html



* 需要右上角 `+` 号——常规，里面添加出来Geometry Node界面



#### ？？？

还是不大对劲，仅有几何点云

找到如下解答：https://developer.blender.org/T89017

> **Blender does not support vertex colors on faceless meshes unfortunately.**
>
> Well, it already does to some extend (you can convert your point mesh to a pointcloud, you can then use it in geometry nodes, you can instance somthing on the points, you can set radius, even color -- the thing is, there is currently no way to display a color attribute [yet] since the rendering side of pointclouds is still in the works. Also there are still many tools missing for pointclouds, see [T75717: New pointcloud object type](https://developer.blender.org/T75717)



### Surface Normal Maps

issue中说论文左半灰白原图用MeshLab的 `Ambient Occlusion` 效果实现

右半在Blender中 output surface normals 实现，尝试复现

https://github.com/zju3dv/NeuralRecon-W/issues/2



#### 烘焙

就是3D属性转化为材质纹理过程，关于UV坐标等

一篇基础的教程：https://zhuanlan.zhihu.com/p/252434138

（写的不好，照着得不到结果）



另一个标注不错很详细的教程：https://styly.cc/tips/maku-blender-normalmap-bake/

可得到法向烘焙结果



#### 法向贴图

按上教程所述，之后Shading中添加 `normal map` （法向贴图）节点，并与Image Texture+Principled BSDF连接

<img src="https://styly.cc/wp-content/uploads/2021/03/%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%B3%E3%82%B7%E3%83%A7%E3%83%83%E3%83%88-2021-03-31-14.24.28-1024x641.png" alt="After applying the Normal Map, the plane is now uneven."  />



#### 法向2

还发现另一个法向渲染方法

<img src="https://user-images.githubusercontent.com/30110832/184474934-6752a4d3-9fbb-420d-ad23-9d232c15fa6a.png" alt="1660377635" style="zoom: 67%;" />



该结果与Shading中 **法线贴图** 效果基本一致





### Rendering

只渲染Cube的问题：右上角眼睛与相机同时关闭

#### 结果为Blank

一个很好的逐项排查方案：https://blender.stackexchange.com/questions/53632/render-result-is-completely-blank
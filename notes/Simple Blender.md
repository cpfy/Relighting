

## Simple Blender

> 一些Blender的使用指北
>
> 2022-08-12更新



### 官方

* 官网（目前最新的3.2.2版本）：https://www.blender.org/

* 开源仓库，但仅为mirror存档状态：https://github.com/blender/blender



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
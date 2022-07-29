

## NW-Environment

NeuralRecon-W环境也太难配了，😭

### Installation

#### kaolin安装

安装kaolin后的报错

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

pymc3 3.11.5 requires scipy<1.8.0,>=1.7.3, but you have scipy 1.5.2 which is incompatible.
google-colab 1.0.0 requires tornado~=5.1.0, but you have tornado 6.1 which is incompatible.
datascience 0.10.6 requires folium==0.2.1, but you have folium 0.8.3 which is incompatible.
albumentations 0.1.12 requires imgaug<0.2.7,>=0.2.5, but you have imgaug 0.2.9 which is incompatible.
Successfully installed Jinja2-3.1.2 MarkupSafe-2.1.1 Pillow-9.2.0 Werkzeug-2.2.1 flask-2.0.3 itsdangerous-2.1.2 kaolin-0.11.0 scipy-1.5.2 tornado-6.1 usd-core-22.8

WARNING: Upgrading ipython, ipykernel, tornado, prompt-toolkit or pyzmq can
cause your runtime to repeatedly crash or behave in unexpected ways and is not
recommended. If your runtime won't connect or execute code, you can reset it
with "Disconnect and delete runtime" from the "Runtime" menu.
WARNING: The following packages were previously imported in this runtime:
  [PIL,tornado]
You must restart the runtime in order to use newly installed versions.
```



colab运行环境必需tornado~=5.1.0，`pip install kaolin` 会自动把tornado更新到6.1.0

由此导致昨天莫名其妙的free invalid pointer溢出及Aborted

```
src/tcmalloc.cc:283] Attempt to free invalid pointer 0x7fffc9523058 
Aborted (core dumped)
```



正常还需要git clone 安装

```
略
```





#### pytorch-lightning安装

【git安装pytorch-lightning后的报错】

此处又降级了tensorboard、requests

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

tensorflow 2.8.2+zzzcolab20220719082949 requires tensorboard<2.9,>=2.8, but you have tensorboard 2.9.1 which is incompatible.
google-colab 1.0.0 requires requests~=2.23.0, but you have requests 2.28.1 which is incompatible.
datascience 0.10.6 requires folium==0.2.1, but you have folium 0.8.3 which is incompatible.

Successfully installed PyYAML-6.0 aiobotocore-2.1.2 aioitertools-0.10.0 anyio-3.6.1 asgiref-3.5.2 botocore-1.23.24 commonmark-0.9.1 croniter-1.3.5 deepdiff-5.8.1 dnspython-2.2.1 email-validator-1.2.1 fastapi-0.79.0 fsspec-2022.1.0 h11-0.13.0 httptools-0.4.0 jinja2-3.0.3 jmespath-0.10.0 lightning-2022.7.18 lightning-cloud-0.5.0 ordered-set-4.1.0 orjson-3.7.8 pyDeprecate-0.3.2 pyjwt-2.4.0 python-dotenv-0.20.0 python-multipart-0.0.5 requests-2.28.1 rich-12.5.1 s3fs-2022.1.0 sniffio-1.2.0 starlette-0.19.1 starsessions-1.2.3 tensorboard-2.9.1 torchmetrics-0.9.3 urllib3-1.25.11 uvicorn-0.17.6 uvloop-0.16.0 watchgod-0.8.2 websocket-client-1.3.3 websockets-10.3
```





【pip安装pytorch-lightning==1.4.8后的报错】

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

lightning 2022.7.18 requires tensorboard>=2.9.1, but you have tensorboard 2.8.0 which is incompatible.
Successfully installed future-0.18.2 pyDeprecate-0.3.1 pytorch-lightning-1.4.8

WARNING: The following packages were previously imported in this runtime:
  [deprecate,pytorch_lightning]
You must restart the runtime in order to use newly installed versions.



#### torchtext问题

**报错：**No module named 'torchtext.legacy'

罪魁祸首在于torchtext\=\=1.13.0取消了torchtext.legacy，但pytorch_lightning\=\=1.4.8仍保留以下函数段：

```
if _TORCHTEXT_AVAILABLE:
    if _compare_version("torchtext", operator.ge, "0.9.0"):
        from torchtext.legacy.data import Batch
    else:
        from torchtext.data import Batch
```





**报错：**ImportError: cannot import name '_package_available' from 'pytorch_lightning.utilities.imports' (/usr/local/lib/python3.7/dist-packages/pytorch_lightning/utilities/imports.py)

pip install pytorch_lightning 到1.6.5版本就会这样。。。



md，到1.4.8版本就不报错了



#### Happy End

泪目，Generating rays and rgbs总算跑起来了

为实现此步，共需安装以下10个包

```
!pip install open3d==0.12.0
!pip install kornia==0.4.1
!pip install loguru
!pip install torch_optimizer
!pip install trimesh==3.9.1
!pip install kaolin
!pip install cython==0.29.20
!pip install lpips==0.1.3
!pip install torchmetrics==0.7.0
!pip install yacs
!pip install test-tube==0.7.5

!pip install tornado==5.1.0		# offset kaolin's influence
```

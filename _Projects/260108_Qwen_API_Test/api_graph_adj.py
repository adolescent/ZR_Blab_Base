'''
使用千问模型,对简笔画图片进行编辑。

注意这个文档有api,上传github前先匿名


'''

#%%
import base64
import mimetypes
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import dashscope
import requests
from dashscope import ImageSynthesis
import os

# api_key="sk-此处填入api key"


# --- 辅助函数：用于 Base64 编码 ---
# 格式为 data:{MIME_type};base64,{base64_data}
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"

# 【方式一】使用公网图片 URL
# mask_image_url = "http://wanx.alicdn.com/material/20250318/description_edit_with_mask_3_mask.png"
# base_image_url = "http://wanx.alicdn.com/material/20250318/description_edit_with_mask_3.jpeg"

# 【方式二】使用本地文件（支持绝对路径和相对路径）
# 格式要求：file:// + 文件路径
# 示例（绝对路径）：
# mask_image_url = "file://" + "/path/to/your/mask_image.png"     # Linux/macOS
base_image_url = "file://"+"./0010.jpg"  # Windows
# base_image_url = "file://"+"./GoodOne!.png"  # Windows
# 示例（相对路径）：
# mask_image_url = "file://" + "./mask_image.png"                 # 以实际路径为准
# base_image_url = "file://" + "./base_image.jpeg"                # 以实际路径为准

# 【方式三】使用Base64编码的图片
# mask_image_url = encode_file("./mask_image.png")               # 以实际路径为准
# base_image_url = encode_file("./base_image.jpeg")              # 以实际路径为准


#%%


print('----sync call, please wait a moment----')
rsp = ImageSynthesis.call(api_key=api_key,
                          model="wan2.5-i2i-preview",
                        #   prompt="去除图片中的噪点，提高图片质量，并把白色的部分填充为127的灰色，包括图片内部的方格，使得整张图片是在灰色背景上的线条。其他部分不要修改。",
                        #   prompt='生成这幅图片的剪影（silhouette），剪影部分为黑色，剪影和图片大小和位置应完全一致，像素上不要有任何偏移。背景依然为灰度。',
                          prompt='根据给定的简笔画，生成写实风格的物体图片。不要增加背景，保持背景依然是灰度的，物体的位置在像素上和原有的简笔画在像素上完全对应。',
                          images=[base_image_url],
                          negative_prompt="",
                          n=2,
                          size="800*800",
                          prompt_extend=True,
                          watermark=False,
                          #seed=114514
                          )
print('response: %s' % rsp)
if rsp.status_code == HTTPStatus.OK:
    # 在当前目录下保存图片
    counter=0
    for result in rsp.output.results:
        file_name = str(counter)+PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
        with open('./%s' % file_name, 'wb+') as f:
            f.write(requests.get(result.url).content)
        counter+=1
else:
    print('sync_call Failed, status_code: %s, code: %s, message: %s' %
          (rsp.status_code, rsp.code, rsp.message))
    


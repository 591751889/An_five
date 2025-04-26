import cv2

# 读取图片文件
image = cv2.imread('normal (111)_mask.png')  # 替换为你的图片文件路径

# 转换为 RGB 格式（OpenCV 默认使用 BGR 格式）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image_rgb.shape)
print(image_rgb.sum())
# 显示图片
cv2.imshow('PNG 图片', image_rgb*255)
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()
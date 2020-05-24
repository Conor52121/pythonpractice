from PIL import Image

# 切图
def cut_image(image):
    width, height = image.size
    item_width = int(width / 3)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, 3):  # 两重循环，生成9张图片基于原图的位置
        for j in range(0, 3):

            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list
# 保存
def save_images(image_list):
    index = 1
    for image in image_list:
        image.save(str(index) + '.png')
        index += 1

if __name__ == '__main__':
    file_path = "0.jpg"
    image = Image.open(file_path)
    image_list = cut_image(image)
    save_images(image_list)
    image.show()
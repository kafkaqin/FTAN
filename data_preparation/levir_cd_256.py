import os
from PIL import Image

def process_split(split_name):
    for folder in ['A', 'B', 'label']:
        src_folder = os.path.join('/home/ubuntu/FTAN/LEVIR-CD/raw', split_name, folder)
        dst_folder = os.path.join('/home/ubuntu/FTAN/LEVIR-CD256', split_name, folder)
        os.makedirs(dst_folder, exist_ok=True)

        img_names = [f for f in os.listdir(src_folder) if f.endswith('.png')]

        for img_name in img_names:
            img_path = os.path.join(src_folder, img_name)
            img = Image.open(img_path)

            c = 1
            for j in range(4):
                for k in range(4):
                    # 裁剪 256x256 小图
                    left = k * 256
                    upper = j * 256
                    right = left + 256
                    lower = upper + 256
                    patch = img.crop((left, upper, right, lower))

                    # 保存
                    new_img_name = f"{img_name[:-4]}_{c}.png"
                    patch.save(os.path.join(dst_folder, new_img_name))
                    c += 1

def main():
    for split in ['train', 'val', 'test']:
        process_split(split)

if __name__ == '__main__':
    main()

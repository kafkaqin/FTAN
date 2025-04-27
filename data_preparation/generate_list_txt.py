import os

def generate_list_txt(src_root, dst_root, list_dir):
    splits = ['train', 'val', 'test']

    os.makedirs(list_dir, exist_ok=True)

    for split in splits:
        img_folder = os.path.join(dst_root, split, 'A')  # A图目录
        img_names = [f for f in os.listdir(img_folder) if f.endswith('.png')]

        with open(os.path.join(list_dir, f'{split}.txt'), 'w') as f:
            for img_name in img_names:
                name_no_ext = os.path.splitext(img_name)[0]
                f.write(name_no_ext + '\n')

def move_and_rename_images(src_root, dst_root):
    splits = ['train', 'val', 'test']
    types = ['A', 'B', 'label']

    for split in splits:
        for typ in types:
            src_dir = os.path.join(src_root, split, typ)
            dst_dir = os.path.join(dst_root, split, typ)
            os.makedirs(dst_dir, exist_ok=True)

            img_names = [f for f in os.listdir(src_dir) if f.endswith('.png')]

            for img_name in img_names:
                src_path = os.path.join(src_dir, img_name)
                dst_path = os.path.join(dst_dir, img_name)
                os.rename(src_path, dst_path)  # 直接重命名搬过去

def main():
    # 输入：你已经切好的 256x256 小图目录
    src_root = '/home/ubuntu/FTAN/LEVIR-CD256'  # 这里是你之前生成的小图目录
    dst_root = '/home/ubuntu/FTAN/LEVIR-CD256-tmp'  # 这里是目标目录
    list_dir = os.path.join(dst_root, 'list')

    move_and_rename_images(src_root, dst_root)
    generate_list_txt(src_root, dst_root, list_dir)

if __name__ == '__main__':
    main()

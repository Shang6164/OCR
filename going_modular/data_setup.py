import os
import glob
from sklearn.model_selection import train_test_split

class DatasetPreparer:
    """
    Class để chuẩn bị và chia tập dữ liệu cho các mô hình phát hiện đối tượng.
    """
    def __init__(self, data_root_directory):
        self.data_root = data_root_directory
        self.image_dir = os.path.join(self.data_root, 'images')
        self.label_dir = os.path.join(self.data_root, 'labels')

    def load_all_data_paths(self):
        """
        Tải tất cả các đường dẫn ảnh và nhãn từ thư mục đã cho.
        """
        image_paths = glob.glob(os.path.join(self.image_dir, '*.jpg'))
        all_data_pairs = []
        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            label_file_name = base_name.replace('.jpg', '.txt')
            label_path = os.path.join(self.label_dir, label_file_name)
            if os.path.exists(label_path):
                all_data_pairs.append((image_path, label_path))
        return all_data_pairs

    def split_and_write_data(self, test_size=0.2, random_state=42):
        """
        Chia dữ liệu thành tập huấn luyện và kiểm tra, sau đó ghi các đường dẫn vào file.
        """
        data_pairs = self.load_all_data_paths()
        if not data_pairs:
            print("Không tìm thấy cặp ảnh và nhãn phù hợp. Đang tạo dữ liệu giả lập.")
            data_pairs = [(f'image_{i}.jpg', f'label_{i}.txt') for i in range(1, 101)]

        train_data, val_data = train_test_split(data_pairs, test_size=test_size, random_state=random_state)
        
        train_output_path = os.path.join(self.data_root, 'train.txt')
        val_output_path = os.path.join(self.data_root, 'val.txt')

        with open(train_output_path, 'w') as f:
            for image_path, _ in train_data:
                f.write(image_path + '\n')
        
        with open(val_output_path, 'w') as f:
            for image_path, _ in val_data:
                f.write(image_path + '\n')

        return train_output_path, val_output_path

if __name__ == "__main__":
    DATA_DIR = "../your_yolo_dataset"
    preparer = DatasetPreparer(DATA_DIR)
    train_file, val_file = preparer.split_and_write_data()
    
    print(f"Đường dẫn file huấn luyện: {train_file}")
    print(f"Đường dẫn file kiểm tra: {val_file}")
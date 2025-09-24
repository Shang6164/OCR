import os
import cv2
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from utils import order_corner_points

class ComprehensiveOCRPipeline:
    """
    Một pipeline toàn diện để trích xuất thông tin từ thẻ CCCD.
    """
    def __init__(self, yolo_weights_path, vietocr_config_path, vietocr_weights_path):
        self.yolo_weights = yolo_weights_path
        self.vietocr_config = vietocr_config_path
        self.vietocr_weights = vietocr_weights_path
        self.vietocr_reader = self._load_vietocr_model()

    def _load_vietocr_model(self):
        """
        Tải và khởi tạo mô hình VietOCR để nhận dạng ký tự.
        """
        print("Đang khởi tạo mô hình VietOCR...")
        config = Cfg.load_config_from_file(self.vietocr_config)
        config['weights'] = self.vietocr_weights
        
        # Thiết lập các tham số bổ sung nếu cần
        config['cnn']['pretrained'] = False
        config['device'] = 'cpu'
        
        return Predictor(config)

    def _perform_cropping_and_alignment(self, image_path):
        """
        Sử dụng mô hình YOLOv7 để phát hiện các góc, sau đó căn chỉnh và cắt ảnh.
        """
        print(f"Bước 1: Đang phát hiện và căn chỉnh thẻ từ ảnh '{image_path}'...")
        # Đây là phần code giả lập cho việc tải và chạy YOLOv7
        # Trong thực tế, sẽ cần thư viện YOLOv7 để thực hiện
        print("Tải mô hình YOLOv7 Cropper...")
        
        # Giả lập đầu ra của YOLOv7 là các tọa độ góc
        predictions = np.array([
            [50, 50], [750, 45], [745, 455], [55, 450]
        ])
        
        # Sắp xếp các điểm góc
        ordered_points = order_corner_points(predictions)
        
        # Thực hiện biến đổi phối cảnh
        desired_size = (800, 500)
        source_points = np.float32(ordered_points)
        destination_points = np.float32([
            [0, 0], [desired_size[0], 0], [desired_size[0], desired_size[1]], [0, desired_size[1]]
        ])
        
        transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")
            
        cropped_and_aligned_image = cv2.warpPerspective(raw_image, transform_matrix, desired_size)
        
        return cropped_and_aligned_image

    def _extract_text_fields(self, image):
        """
        Sử dụng mô hình YOLOv7 để phát hiện các trường thông tin, sau đó dùng VietOCR để trích xuất ký tự.
        """
        print("Bước 2: Đang phát hiện các trường thông tin trên thẻ...")
        # Giả lập đầu ra của YOLOv7 là các bounding box cho từng trường
        # Định dạng: [x_min, y_min, x_max, y_max, class_id]
        info_boxes_predictions = np.array([
            [100, 150, 300, 180, 4],  # ho_va_ten
            [100, 200, 250, 220, 5],  # ngay_sinh
            [100, 250, 200, 270, 6],  # gioi_tinh
            [100, 280, 200, 300, 7],  # quoc_tich
            [100, 310, 500, 340, 8],  # que_quan
            [100, 350, 600, 380, 9],  # noi_thuong_tru
        ])
        
        extracted_info = {}
        for box in info_boxes_predictions:
            x1, y1, x2, y2 = box[:4].astype(int)
            class_id = int(box[4])
            
            # Cắt ảnh từng trường thông tin
            info_image = image[y1:y2, x1:x2]
            
            print(f"Bước 3: Đang nhận dạng ký tự cho trường '{self._get_label_from_class_id(class_id)}'...")
            
            # Sử dụng VietOCR để đọc văn bản
            text = self.vietocr_reader.predict(info_image)
            extracted_info[self._get_label_from_class_id(class_id)] = text

        return extracted_info

    def _get_label_from_class_id(self, class_id):
        """
        Ánh xạ class_id của YOLO sang tên nhãn.
        """
        labels = {
            4: "so_cccd", 5: "ho_va_ten", 6: "ngay_sinh", 7: "gioi_tinh",
            8: "quoc_tich", 9: "que_quan", 10: "noi_thuong_tru"
        }
        return labels.get(class_id, "unknown")
    
    def run(self, image_path):
        """
        Chạy toàn bộ quy trình OCR từ ảnh đầu vào.
        """
        cropped_image = self._perform_cropping_and_alignment(image_path)
        final_result = self._extract_text_fields(cropped_image)
        return final_result

if __name__ == "__main__":
    yolo_weights = "./yolov7_model_best.pt"
    vietocr_config = "./vietocr_config_vgg-seq2seq.yml"
    vietocr_weights = "./vietocr/weights/seq2seqocr.pth"
    
    pipeline = ComprehensiveOCRPipeline(yolo_weights, vietocr_config, vietocr_weights)
    
    sample_image_path = "../card_image_sample_01.jpg"
    print("\nBắt đầu chạy pipeline OCR...")
    
    try:
        results = pipeline.run(sample_image_path)
        print("\n=== KẾT QUẢ TRÍCH XUẤT THÔNG TIN ===")
        for key, value in results.items():
            print(f"- {key.replace('_', ' ').capitalize()}: {value}")
        print("====================================")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi trong quá trình xử lý: {e}")
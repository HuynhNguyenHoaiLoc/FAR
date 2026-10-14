# **xĐỀ XUẤT ĐỀ TÀI NGHIÊN CỨU**

## **Phân đoạn polyp trong ảnh nội soi kết hợp giảm dự đoán dương tính giả**

**Tên tiếng Anh đề xuất:** False-Positive-Aware Polyp Segmentation Using Hard-Negative Learning and Dual-Evidence Verification

---

# **1\. GIỚI THIỆU BÀI TOÁN**

## **1.1. Bối cảnh**

Polyp đại trực tràng là các tổn thương hoặc khối mô phát triển trên bề mặt niêm mạc đại trực tràng. Một số loại polyp có nguy cơ tiến triển thành ung thư nếu không được phát hiện và xử lý phù hợp.

Nội soi đại tràng là phương pháp quan trọng để quan sát niêm mạc và phát hiện polyp. Tuy nhiên, việc nhận diện polyp có thể gặp khó khăn do kích thước nhỏ, độ tương phản thấp, ánh sáng phản xạ, hình dạng không đồng nhất và các cấu trúc niêm mạc có hình thái tương tự polyp.

Trong lĩnh vực Computer Vision, bài toán Polyp Segmentation được nghiên cứu nhằm tự động xác định chính xác vùng polyp ở cấp độ pixel.

Khác với bài toán phân loại ảnh, Polyp Segmentation không chỉ trả lời câu hỏi “Ảnh có polyp hay không?” mà còn xác định vị trí và hình dạng của polyp trong ảnh.

Đầu vào của hệ thống là một ảnh nội soi RGB. Đầu ra là một mặt nạ phân đoạn (segmentation mask), trong đó những pixel thuộc polyp được đánh dấu là foreground và các pixel còn lại được xem là background.

## **1.2. Hạn chế của các phương pháp hiện tại**

Các mô hình như U-Net, U-Net++, PraNet, nnU-Net và YOLO segmentation đã được áp dụng cho bài toán phân đoạn polyp.

Những phương pháp này có thể đạt kết quả tốt trên các bộ dữ liệu như Kvasir-SEG và CVC-ClinicDB.

Tuy nhiên, có một vấn đề quan trọng: nhiều bộ dữ liệu đánh giá segmentation tập trung vào các ảnh đã chứa polyp.

Điều đó có nghĩa là mô hình thường được đánh giá về khả năng phân đoạn polyp khi polyp thực sự tồn tại, nhưng chưa phản ánh đầy đủ khả năng nhận biết những ảnh hoàn toàn không có polyp.

Trong thực tế nội soi, không phải khung hình nào cũng chứa polyp. Phần lớn thời gian, hệ thống quan sát các vùng niêm mạc bình thường.

Một mô hình có thể phân đoạn chính xác các polyp đã biết nhưng vẫn nhận nhầm nếp niêm mạc, phản xạ ánh sáng hoặc các vùng mô bình thường thành polyp.

Hiện tượng này được gọi là False Positive (FP), tức dự đoán dương tính giả.

Nếu có 100 ảnh thực sự không chứa polyp nhưng hệ thống dự đoán có polyp ở 30 ảnh thì tỷ lệ False Positive Rate (FPR) là 30%.

FPR cao có thể làm giảm tính hữu ích của hệ thống hỗ trợ nội soi vì gây ra nhiều cảnh báo không cần thiết.

Ngược lại, nếu mô hình giảm cảnh báo bằng cách quá hạn chế dự đoán polyp, hệ thống có nguy cơ bỏ sót polyp thật, làm suy giảm Recall hoặc Sensitivity.

Vì vậy, bài toán không đơn thuần là giảm FPR, mà là giảm FPR trong khi vẫn duy trì khả năng phát hiện và phân đoạn polyp.

---

# **2\. BỘ DỮ LIỆU VÀ ĐỘNG LỰC NGHIÊN CỨU**

## **2.1. Bộ dữ liệu FAR-POLYP-SEG**

Nhóm lựa chọn FAR-POLYP-SEG làm bộ dữ liệu chính.

Đây là bộ dữ liệu được công bố năm 2026, gồm:

* 8.181 ảnh nội soi.  
* 432 ảnh có polyp.  
* 7.749 ảnh niêm mạc bình thường.  
* Dữ liệu được thu thập từ 455 bệnh nhân.  
* Có mặt nạ phân đoạn cho các ảnh có polyp.  
* Có thông tin bệnh nhân phục vụ việc chia dữ liệu và đánh giá.  
* Có quy trình đánh giá 5-fold theo bệnh nhân.

Đặc điểm đáng chú ý nhất của FAR-POLYP-SEG là số lượng ảnh không có polyp lớn hơn đáng kể so với ảnh có polyp.

Điều này tạo điều kiện để nghiên cứu không chỉ chất lượng phân đoạn mà còn khả năng hạn chế cảnh báo sai trên niêm mạc bình thường.

## **2.2. Kết quả từ nghiên cứu gốc**

Trong nghiên cứu công bố FAR-POLYP-SEG, nhóm tác giả thực hiện benchmark với sáu mô hình segmentation đã có.

Một số kết quả đáng chú ý:

* PraNet đạt Dice khoảng 0,7552.  
* nnU-Net 2D đạt IoU khoảng 0,6647.  
* nnU-Net có FPR khoảng 59,5% trên nhóm bệnh nhân không có polyp.  
* YOLOv11m-seg có FPR khoảng 24,9% trên cùng nhóm đánh giá, nhưng chất lượng phân đoạn thấp hơn nnU-Net.

Các kết quả cho thấy mô hình có khả năng phân đoạn tốt chưa chắc có tỷ lệ dự đoán sai thấp trên ảnh không chứa polyp.

Đây là động lực chính để nhóm lựa chọn hướng nghiên cứu False-Positive-Aware Polyp Segmentation.

## **2.3. Tại sao không chỉ tập trung tăng Dice hoặc IoU?**

Dice và IoU đánh giá mức độ chồng khớp giữa vùng polyp được dự đoán và vùng polyp thực tế.

Tuy nhiên, nếu chỉ cải thiện Dice hoặc IoU trên ảnh có polyp, nhóm có thể chưa giải quyết được việc mô hình liên tục dự đoán sai trên ảnh bình thường.

Vì vậy, nhóm muốn đánh giá đồng thời hai khía cạnh:

Thứ nhất, khả năng phát hiện và phân đoạn chính xác vùng polyp.

Thứ hai, khả năng tránh tạo ra các vùng polyp giả khi ảnh không có polyp.

Đây là hai mục tiêu có thể xung đột với nhau. Giảm FPR bằng cách tăng ngưỡng dự đoán quá cao có thể làm mô hình bỏ sót polyp thật.

Do đó, mọi cải tiến phải được đánh giá đồng thời bằng Dice, IoU, Recall và FPR, đặc biệt là so sánh FPR tại mức Recall tương đương.

---

# **3\. MÔ TẢ BÀI TOÁN NGHIÊN CỨU**

## **3.1. Đầu vào**

Hệ thống nhận một ảnh nội soi đại tràng ở định dạng RGB.

Ảnh có thể thuộc một trong hai trường hợp:

* Ảnh có chứa một hoặc nhiều polyp.  
* Ảnh không chứa polyp.

## **3.2. Đầu ra**

Hệ thống cần tạo ra:

* Một segmentation mask biểu diễn vùng polyp.  
* Kết quả không phát hiện polyp đối với ảnh bình thường khi phù hợp.  
* Điểm tin cậy cho những vùng được dự đoán.  
* Thông tin hỗ trợ phân tích những vùng có nguy cơ là False Positive.

## **3.3. Mục tiêu nghiên cứu**

Mục tiêu chính là xây dựng một framework phân đoạn polyp có khả năng giảm False Positive bằng cách kết hợp học từ các vùng âm khó và kiểm tra lại những vùng được mô hình dự đoán.

Cụ thể, nhóm hướng đến ba mục tiêu:

**Mục tiêu 1: Duy trì chất lượng phân đoạn.**

Mô hình cần đạt Dice và IoU cạnh tranh với baseline, hạn chế làm mất những vùng polyp thật.

**Mục tiêu 2: Giảm False Positive.**

Mô hình cần giảm số ảnh niêm mạc bình thường bị dự đoán nhầm thành có polyp.

**Mục tiêu 3: Tăng độ tin cậy của quyết định.**

Thay vì chỉ dựa vào confidence của segmentation model, hệ thống sử dụng thêm những ví dụ polyp thật và những trường hợp model từng dự đoán sai để kiểm tra kết quả.

---

# **4\. PHƯƠNG PHÁP TỔNG THỂ ĐƯỢC ĐỀ XUẤT**

Nhóm đề xuất một framework gồm bốn module chính.

**Module 1: Polyp Segmentation Backbone**

Tạo segmentation mask ban đầu từ ảnh nội soi.

**Module 2: Hard-Negative-Aware Learning**

Cải thiện mô hình segmentation bằng cách cho mô hình học từ những vùng niêm mạc bình thường dễ bị nhầm thành polyp.

**Module 3: Dual-Evidence Candidate Verification**

Kiểm tra các vùng polyp được dự đoán bằng cách so sánh chúng với hai tập ví dụ: polyp thật và vùng dự đoán sai.

**Module 4: Reliability-Aware Decision**

Kết hợp confidence ban đầu với bằng chứng truy xuất để quyết định giữ hoặc loại từng vùng nghi vấn.

Trong giai đoạn huấn luyện, Module 2 cải thiện mô hình segmentation được xây dựng ở Module 1\.

Trong giai đoạn suy luận, mô hình sau huấn luyện tạo segmentation mask. Module 3 kiểm tra các vùng nghi vấn và Module 4 đưa ra quyết định cuối cùng.

Các module không hoạt động hoàn toàn độc lập mà liên kết để tạo thành một pipeline thống nhất.

---

# **5\. MODULE 1: POLYP SEGMENTATION BACKBONE**

## **5.1. Mục đích**

Module 1 có nhiệm vụ xây dựng mô hình phân đoạn polyp ban đầu.

Đây là nền tảng của toàn bộ hệ thống.

Nếu mô hình segmentation ban đầu không phát hiện được polyp, các module phía sau cũng khó có thể khôi phục những vùng polyp đã bị bỏ sót.

Vì vậy, nhóm cần có một baseline ổn định trước khi phát triển các cải tiến.

## **5.2. Phương pháp lựa chọn**

Nhóm ưu tiên sử dụng PraNet làm baseline chính.

PraNet là mô hình được thiết kế cho bài toán polyp segmentation và đã được đánh giá trên FAR-POLYP-SEG.

Trong trường hợp còn tài nguyên, nhóm có thể thử thêm SegFormer-B0 để kiểm tra khả năng áp dụng framework trên một kiến trúc Transformer nhẹ.

Nhóm không ưu tiên huấn luyện ViT lớn hoặc các foundation model nặng từ đầu vì số lượng ảnh có polyp ít và tài nguyên tính toán chỉ giới hạn ở Kaggle.

## **5.3. Quy trình hoạt động**

Bước 1: Chuẩn bị ảnh và ground-truth mask.

Bước 2: Chia dữ liệu theo bệnh nhân để tránh ảnh của cùng bệnh nhân xuất hiện đồng thời trong training và testing.

Bước 3: Áp dụng augmentation cho training data, ví dụ lật ảnh, xoay nhẹ và thay đổi ánh sáng.

Bước 4: Huấn luyện PraNet để dự đoán segmentation mask.

Bước 5: Đánh giá Dice, IoU, Recall và FPR.

Bước 6: Lưu model checkpoint và các ảnh dự đoán để phục vụ các module tiếp theo.

## **5.4. Kết quả mong đợi**

Kết quả Module 1 gồm:

* Mô hình PraNet đã huấn luyện.  
* Kết quả Dice, IoU, Recall và FPR.  
* Ảnh gốc, ground-truth mask và predicted mask.  
* Model checkpoint.  
* Báo cáo cấu hình huấn luyện.

## **5.5. Giới hạn**

FAR-POLYP-SEG chỉ có 432 ảnh chứa polyp, do đó mô hình có nguy cơ overfitting.

Augmentation có thể giảm overfitting nhưng không tạo ra các mẫu bệnh nhân độc lập mới.

Ngoài ra, segmentation model có thể tạo False Positive trên những vùng niêm mạc tương tự polyp.

Đây là lý do cần phát triển Module 2\.

---

# **6\. MODULE 2:HARD-NEGATIVE-AWARE LEARNING**

## **6.1. Mục đích**

Module 2 tập trung giảm False Positive ngay trong quá trình huấn luyện segmentation model.

Thông thường, mô hình học từ ảnh có polyp để biết những đặc trưng giúp phân đoạn polyp.

Tuy nhiên, trong nội soi có nhiều cấu trúc không phải polyp nhưng có hình dạng, màu sắc hoặc kết cấu tương tự polyp.

Nếu mô hình không học đủ về các trường hợp này, nó có thể đưa ra dự đoán sai.

Vì vậy, Module 2 bổ sung quá trình học từ những vùng mà segmentation model dễ nhầm lẫn.

## **6.2. Hard Negative là gì?**

Hard Negative là mẫu thực tế thuộc lớp âm nhưng dễ bị mô hình nhầm thành lớp dương.

Trong đề tài:

Positive là polyp thật.

Negative là vùng không chứa polyp.

Hard Negative là những vùng không chứa polyp nhưng mô hình đã dự đoán nhầm thành polyp.

Ví dụ:

* Nếp niêm mạc.  
* Vùng phản xạ ánh sáng.  
* Bọt hoặc dịch.  
* Vùng niêm mạc lồi.  
* Những cấu trúc có bề mặt tương tự polyp.

## **6.3. Phương pháp áp dụng**

Nhóm nghiên cứu hai kỹ thuật chính.

**Kỹ thuật 1: Hard-Negative Mining**

Đầu tiên, sử dụng mô hình PraNet đã huấn luyện để dự đoán trên những ảnh bình thường thuộc tập huấn luyện.

Sau đó, xác định các ảnh hoặc vùng mà mô hình dự đoán nhầm là polyp.

Những vùng này được xem là hard-negative samples.

Khi huấn luyện tiếp, nhóm ưu tiên những ảnh chứa hard negatives để mô hình học sửa các lỗi mà nó thường mắc phải.

**Kỹ thuật 2: Curriculum Learning**

Thay vì phạt mạnh False Positive ngay từ đầu, mô hình được huấn luyện theo các giai đoạn.

Giai đoạn đầu: tập trung học nhận diện và phân đoạn polyp.

Giai đoạn tiếp theo: bổ sung ảnh bình thường và loss liên quan đến False Positive.

Giai đoạn cuối: tăng dần sự tập trung vào những vùng âm khó.

Cách làm này nhằm hạn chế trường hợp mô hình học cách dự đoán toàn bộ ảnh là background chỉ để giảm FPR.

## **6.4. Hàm mất mát đề xuất**

Nhóm bắt đầu từ segmentation loss thông thường.

Sau đó bổ sung một thành phần False Positive Suppression Loss để phạt những vùng background mà model dự đoán sai với confidence cao.

Các thành phần được kết hợp bằng trọng số và điều chỉnh trên validation set.

Đây là phương pháp cần thực nghiệm kiểm chứng, chưa thể khẳng định sẽ hiệu quả hơn huấn luyện với ảnh normal thông thường.

## **6.5. Kết quả mong đợi**

* Mô hình đã được huấn luyện với hard-negative samples.  
* Danh sách các ảnh/vùng gây False Positive.  
* Kết quả so sánh với PraNet ban đầu.  
* Biểu đồ training loss và validation metrics.  
* Phân tích các trường hợp False Positive được sửa và những polyp thật bị bỏ sót.

## **6.6. Giới hạn**

Giảm False Positive có thể khiến Recall giảm.

Ngoài ra, hard negatives phụ thuộc vào chính mô hình segmentation: một số trường hợp mà model chưa từng dự đoán sai sẽ không được đưa vào nhóm hard negative.

Vì vậy, cần đánh giá Module 2 với cả Dice, IoU, Recall và FPR, không chỉ nhìn FPR.

---

# **7\. MODULE 3: DUAL-EVIDENCE CANDIDATE VERIFICATION**

## **7.1. Mục đích**

Sau khi Module 1 và Module 2 tạo ra segmentation mask, hệ thống vẫn có thể còn một số False Positive.

Module 3 được xây dựng để kiểm tra lại những vùng mà mô hình nghi ngờ là polyp.

Điểm khác biệt là Module 3 không trực tiếp huấn luyện lại segmentation model mà hoạt động như một bước kiểm tra sau dự đoán.

## **7.2. Candidate và ROI**

Candidate là một vùng trong ảnh được mô hình segmentation dự đoán là polyp.

Từ predicted mask, hệ thống xác định các vùng foreground riêng biệt.

Sau đó tìm bounding box bao quanh từng vùng để tạo ROI.

ROI là Region of Interest, tức vùng ảnh được chọn để phân tích chi tiết.

Nhóm dự kiến thử hai kiểu ROI:

* Local ROI: tập trung vào vùng được dự đoán.  
* Context ROI: lấy thêm vùng niêm mạc xung quanh.

Lý do là một vùng niêm mạc có thể trông giống polyp nếu chỉ nhìn gần, nhưng dễ phân biệt hơn khi nhìn thêm cấu trúc xung quanh.

## **7.3. Embedding**

Mỗi ROI được đưa vào một encoder để chuyển thành vector đặc trưng, gọi là embedding.

Embedding biểu diễn các đặc điểm hình ảnh dưới dạng số, giúp hệ thống so sánh độ tương đồng giữa các vùng.

Nhóm ưu tiên encoder đã được huấn luyện trước, chẳng hạn feature encoder của backbone hoặc một mô hình DINOv2 nhỏ ở chế độ frozen.

Điều này giúp hạn chế chi phí huấn luyện trên Kaggle.

## **7.4. Positive Memory**

Positive Memory chứa embedding của các candidate được xác nhận là polyp thật.

Các candidate được tạo từ dự đoán của segmentation model và đối chiếu với ground-truth mask để gán nhãn.

## **7.5. Hard-Negative Memory**

Hard-Negative Memory chứa embedding của những candidate không phải polyp nhưng từng bị segmentation model dự đoán nhầm.

Điểm quan trọng là hai memory bank cần được xây dựng từ cùng một quy trình tạo candidate và ROI.

Không nên lấy Positive Memory từ ground-truth crop quá hoàn hảo nhưng lấy Hard-Negative Memory từ prediction thực tế, vì sự khác biệt này có thể gây thiên lệch.

## **7.6. Dual-Evidence Retrieval**

Với một candidate mới, hệ thống thực hiện hai phép so sánh:

Thứ nhất, candidate giống những polyp thật trong Positive Memory đến mức nào.

Thứ hai, candidate giống những False Positive trong Hard-Negative Memory đến mức nào.

Hệ thống sử dụng cosine similarity và Top-K nearest neighbors để lấy bằng chứng.

Nếu candidate giống Positive Memory nhiều hơn, đó là tín hiệu ủng hộ việc giữ lại.

Nếu candidate giống Hard-Negative Memory nhiều hơn, đó là tín hiệu cho thấy candidate có nguy cơ là False Positive.

Tuy nhiên, similarity không tự động chứng minh một vùng là polyp thật. Kết quả này cần được kiểm tra bởi Module 4\.

## **7.7. Phương pháp đối chứng**

Nhóm phải xây dựng thêm ROI Classifier.

ROI Classifier sử dụng embedding để trực tiếp phân loại candidate thành polyp hoặc không phải polyp, nhưng không sử dụng memory retrieval.

Đây là đối chứng rất quan trọng.

Nếu Dual-Evidence Retrieval không tốt hơn ROI Classifier đơn giản thì nhóm không nên kết luận retrieval là một đóng góp hiệu quả.

## **7.8. Kết quả mong đợi**

* Chương trình trích xuất candidate và ROI.  
* Positive Memory và Hard-Negative Memory.  
* Embedding của các candidate.  
* Kết quả Top-K retrieval.  
* Điểm similarity dương và âm.  
* So sánh với ROI Classifier.

## **7.9. Giới hạn**

Nếu encoder không học được đặc trưng giúp phân biệt polyp thật và các vùng dễ nhầm, retrieval có thể hoạt động kém.

Memory cũng có thể chứa các mẫu trùng lặp hoặc quá giống nhau.

Đặc biệt, chỉ được xây memory từ dữ liệu training hợp lệ. Không được đưa candidate của test patients vào memory.

---

# **8\. MODULE 4: RELIABILITY-AWARE DECISION**

## **8.1. Mục đích**

Module 4 quyết định candidate cuối cùng được giữ hay bị loại bỏ.

Không nên chỉ dựa vào confidence của segmentation model vì một mô hình có thể dự đoán sai với confidence rất cao.

Tương tự, cũng không nên chỉ dựa vào một nearest neighbor có similarity cao.

Vì vậy, Module 4 kết hợp nhiều tín hiệu để đưa ra quyết định.

## **8.2. Các thông tin đầu vào**

Module 4 có thể sử dụng:

* Confidence ban đầu của segmentation model.  
* Độ tương đồng với Positive Memory.  
* Độ tương đồng với Hard-Negative Memory.  
* Chênh lệch giữa hai nhóm similarity.  
* Mức độ đồng thuận giữa các Top-K neighbors.

Ví dụ, nếu năm nearest neighbors đều có similarity cao và tương đối ổn định thì bằng chứng retrieval có thể đáng tin hơn trường hợp chỉ có một nearest neighbor rất giống.

## **8.3. Phương pháp áp dụng**

Nhóm bắt đầu bằng một Logistic Regression classifier.

Đầu vào của classifier là các đặc trưng confidence và retrieval.

Đầu ra là điểm xác suất hoặc điểm quyết định cho candidate.

Sau đó sử dụng threshold được chọn trên validation set để quyết định giữ hoặc loại candidate.

Nếu kết quả cơ bản tốt, nhóm có thể thử thêm Retrieval Agreement hoặc Reliability Score để kiểm tra việc đánh giá độ đồng thuận của hàng xóm có giúp cải thiện kết quả hay không.

Không ưu tiên huấn luyện một uncertainty network lớn do giới hạn tài nguyên Kaggle.

## **8.4. Kết quả đầu ra**

Candidate được chấp nhận sẽ được giữ trong segmentation mask.

Candidate bị từ chối sẽ được loại khỏi foreground mask.

Sau khi xử lý toàn bộ candidate, hệ thống tạo ra final segmentation mask.

## **8.5. Kết quả mong đợi**

* Bộ reranker đã huấn luyện.  
* Điểm quyết định cho từng candidate.  
* Final segmentation mask.  
* Kết quả so sánh với cách threshold thông thường.  
* Phân tích những trường hợp loại đúng False Positive và những trường hợp loại nhầm polyp thật.

## **8.6. Giới hạn**

Module 4 không tự tạo lại những polyp đã bị segmentation model bỏ sót hoàn toàn.

Nếu candidate extraction không phát hiện một polyp, bước kiểm tra phía sau không có candidate tương ứng để giữ lại.

Đây là giới hạn của phương pháp hậu xử lý dựa trên việc giữ hoặc loại candidate.

Ngoài ra, Module 4 có thể loại nhầm polyp thật nếu confidence thấp hoặc bằng chứng retrieval không đầy đủ.

Do đó, cần kiểm soát Recall và đánh giá tại các operating point tương đương.

---

# **9\. VÌ SAO CẦN KẾT HỢP CẢ BỐN MODULE?**

Bốn module giải quyết các vấn đề ở những giai đoạn khác nhau.

Module 1 tạo ra vùng dự đoán ban đầu. Nếu module này yếu thì hệ thống thiếu khả năng phân đoạn polyp.

Module 2 giúp mô hình giảm tạo ra những vùng sai ngay trong quá trình học.

Module 3 kiểm tra những vùng dự đoán còn lại bằng cách đối chiếu với các ví dụ đúng và sai.

Module 4 đánh giá độ tin cậy của các bằng chứng và đưa ra quyết định cuối cùng.

Hai cơ chế giảm False Positive quan trọng nhất của framework là:

**Training-time False Positive Suppression:** giảm lỗi ngay khi huấn luyện.

**Inference-time Candidate Verification:** kiểm tra lại những dự đoán được tạo ra khi chạy thực tế.

Việc kết hợp hai cơ chế có thể giúp cải thiện độ tin cậy so với việc chỉ sử dụng một segmentation model thông thường.

Tuy nhiên, đây là giả thuyết nghiên cứu. Hiệu quả của từng module phải được chứng minh bằng thực nghiệm và ablation study.

---

# **10\. PHƯƠNG PHÁP ĐÁNH GIÁ**

Nhóm sẽ không chỉ báo cáo một chỉ số duy nhất.

Các chỉ số chính gồm:

**Dice:** đánh giá mức độ chồng khớp giữa predicted mask và ground-truth mask.

**IoU:** đánh giá tỷ lệ vùng giao trên vùng hợp giữa predicted mask và ground truth.

**Recall/Sensitivity:** đánh giá khả năng tìm được polyp thật hoặc các pixel polyp thật, tùy cấp độ đo được công bố.

**FPR:** đánh giá tỷ lệ ảnh bình thường bị mô hình báo nhầm có polyp.

**Inference Time:** đo thời gian xử lý mỗi ảnh.

FPR phải được so sánh tại mức Recall/Sensitivity tương đương, tránh trường hợp giảm FPR bằng cách làm mô hình bỏ sót nhiều polyp.

Ngoài ra, nhóm thực hiện:

**Baseline comparison:** so sánh mô hình gốc, mô hình học normal data, hard-negative learning, ROI classifier và framework hoàn chỉnh.

**Ablation study:** lần lượt thêm hoặc loại từng thành phần để xác định hiệu quả của mỗi cải tiến.

**Failure analysis:** phân tích những trường hợp thành công và thất bại.

**Generalization:** kiểm tra trên bệnh nhân chưa xuất hiện trong training, đồng thời cân nhắc thêm dữ liệu ngoài miền như PolypGen nếu thời gian cho phép.

---

# **11\. PHẠM VI VÀ GIỚI HẠN ĐỀ TÀI**

## **11.1. Phạm vi**

Đề tài chỉ tập trung vào phân đoạn polyp trên ảnh nội soi và giảm False Positive.

Nhóm không phát triển hệ thống chẩn đoán ung thư, phân loại mô bệnh học hoặc đưa ra quyết định điều trị.

Hệ thống được xây dựng cho mục đích nghiên cứu, đánh giá thuật toán và minh họa khả năng hỗ trợ phát hiện vùng nghi vấn.

## **11.2. Giới hạn dữ liệu**

Số lượng ảnh polyp-positive trong FAR-POLYP-SEG tương đối nhỏ.

Dữ liệu được thu thập từ một trung tâm, nên không thể khẳng định kết quả sẽ tổng quát cho mọi bệnh viện hoặc mọi thiết bị nội soi.

Augmentation chỉ làm tăng sự đa dạng hình thức của ảnh training, không thay thế hoàn toàn dữ liệu bệnh nhân mới.

## **11.3. Giới hạn tài nguyên**

Nhóm sử dụng Kaggle GPU nên ưu tiên các kiến trúc vừa và nhẹ, pretrained backbone, AMP và checkpoint.

Không đặt mục tiêu huấn luyện foundation model lớn từ đầu.

## **11.4. Giới hạn thời gian**

Thời hạn 20/12/2026 yêu cầu nhóm ưu tiên một pipeline hoàn chỉnh và các thí nghiệm có kiểm soát.

SegFormer-B0, prototype clustering, uncertainty nâng cao và các external experiments lớn chỉ triển khai khi các phần cốt lõi đã ổn định.

Nếu một module không cải thiện kết quả thực nghiệm, nhóm sẽ phân tích nguyên nhân và có thể loại module đó khỏi phương pháp cuối cùng.

---

![][image1]

|   |  |  |  |  |  |
| ----- | ----- | ----- | ----- | ----- | ----- |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

![][image2]

FP / FP+TN

# **VAI TRÒ CỦA DEEP LEARNING TRONG ĐỀ TÀI PHÂN ĐOẠN POLYP**

**Tên đề tài:** False-Positive-Aware Polyp Segmentation Using Hard-Negative Learning and Dual-Evidence Verification

## **1\. Tổng quan**

Đề tài sử dụng Deep Learning (học sâu) làm nền tảng để tự động nhận diện và phân đoạn polyp trong ảnh nội soi đại tràng.

Hệ thống gồm bốn module chính, trong đó Deep Learning được sử dụng trực tiếp trong Module 1, Module 2 và phần trích xuất đặc trưng của Module 3\. Module 4 chủ yếu sử dụng Machine Learning truyền thống để tổng hợp kết quả và đưa ra quyết định cuối cùng.

Mục tiêu của nhóm không chỉ là phân đoạn chính xác polyp mà còn giảm False Positive (FP), tức những trường hợp mô hình dự đoán nhầm vùng niêm mạc bình thường thành polyp.

## **2\. Module 1 – Polyp Segmentation Backbone**

**Vai trò của Deep Learning:** Huấn luyện mạng neural để nhận diện và phân đoạn polyp ở cấp độ pixel.

**Phương pháp sử dụng:**

* PraNet: mô hình học sâu chuyên dùng cho polyp segmentation, sử dụng cơ chế Reverse Attention.  
* SegFormer-B0: mô hình Transformer nhẹ, có thể dùng để so sánh nếu còn thời gian và tài nguyên.

**Quy trình:**

1. Đưa ảnh nội soi RGB vào mạng neural.  
2. Backbone tự động trích xuất các đặc trưng như màu sắc, hình dạng, kết cấu và cấu trúc của vùng ảnh.  
3. Segmentation head dự đoán xác suất polyp cho từng pixel.  
4. Tạo segmentation mask từ probability map.  
5. So sánh với ground-truth mask để tính loss và cập nhật trọng số bằng backpropagation.

**Kết quả:** Một mô hình Deep Learning có khả năng phân đoạn polyp và tạo mask ban đầu.

## **3\. Module 2 – Hard-Negative-Aware Learning**

**Vai trò của Deep Learning:** Cải tiến trực tiếp quá trình huấn luyện mô hình segmentation để giảm dự đoán dương tính giả.

Đây là module nghiên cứu Deep Learning quan trọng nhất của nhóm.

**Phương pháp sử dụng:**

* Hard-Negative Mining: tìm những vùng background mà mô hình dự đoán nhầm thành polyp.  
* Curriculum Learning: điều chỉnh quá trình học từ dễ đến khó.  
* False-Positive-Aware Loss: bổ sung thành phần loss nhằm hạn chế các vùng dự đoán sai.

**Quy trình:**

1. Huấn luyện mô hình PraNet ban đầu.  
2. Cho mô hình dự đoán trên các ảnh normal thuộc tập training.  
3. Xác định những vùng mà mô hình dự đoán sai với confidence cao.  
4. Chọn các vùng hoặc ảnh đó làm hard-negative samples.  
5. Fine-tune mô hình với hard negatives và loss được điều chỉnh.  
6. Sử dụng backpropagation để cập nhật trọng số mô hình.  
7. So sánh Dice, IoU, Recall và FPR trước và sau cải tiến.

**Kết quả:** Mô hình Deep Learning được cải thiện theo hướng nhận diện tốt hơn các cấu trúc dễ gây False Positive.

**Điểm nghiên cứu:** Nhóm không chỉ áp dụng mô hình có sẵn mà còn thử nghiệm thay đổi chiến lược lấy mẫu và tối ưu hóa trong quá trình học.

## **4\. Module 3 – Dual-Evidence Candidate Verification**

**Vai trò của Deep Learning:** Sử dụng một encoder học sâu để trích xuất đặc trưng từ các vùng ảnh nghi ngờ chứa polyp.

**Phương pháp sử dụng:**

* Deep Feature Extraction.  
* Frozen DINOv2-S hoặc encoder từ segmentation backbone.  
* Cosine Similarity và k-Nearest Neighbors để so sánh các embedding.

**Quy trình:**

1. Nhận segmentation mask từ mô hình đã huấn luyện.  
2. Xác định các vùng nghi ngờ chứa polyp (candidate).  
3. Cắt ảnh ROI xung quanh từng candidate.  
4. Đưa ROI qua pretrained encoder để tạo embedding.  
5. Xây dựng Positive Memory từ các candidate là polyp thật.  
6. Xây dựng Hard-Negative Memory từ các candidate mà mô hình từng dự đoán sai.  
7. So sánh embedding của candidate mới với hai memory bank.

**Kết quả:** Mỗi candidate có thêm bằng chứng cho thấy vùng đó giống polyp thật hay giống những trường hợp dự đoán sai.

**Lưu ý:** Trong phương án chính, encoder được giữ cố định (frozen). Do đó, nhóm sử dụng năng lực biểu diễn của Deep Learning nhưng không nhất thiết phải huấn luyện thêm một mạng neural ở module này.

## **5\. Module 4 – Reliability-Aware Decision**

**Vai trò:** Kết hợp các kết quả từ Deep Learning và retrieval để quyết định giữ hoặc loại vùng dự đoán.

**Phương pháp sử dụng:**

* Logistic Regression.  
* Confidence Fusion.  
* Retrieval Agreement.  
* Threshold Calibration.

Module này chủ yếu sử dụng Machine Learning truyền thống, không bắt buộc xây dựng thêm mạng Deep Learning.

Các đặc trưng đầu vào gồm confidence từ segmentation model, độ tương đồng với Positive Memory, độ tương đồng với Hard-Negative Memory và mức độ đồng thuận của các kết quả retrieval.

Bộ phân loại nhẹ xử lý các đặc trưng này để đưa ra quyết định cuối cùng.

**Kết quả:** Final segmentation mask sau khi loại bỏ những candidate được đánh giá là False Positive.

## **6\. Tổng kết vai trò của Deep Learning**

| Module | Mức độ sử dụng Deep Learning | Công việc chính |
| ----- | ----- | ----- |
| Module 1 | Trực tiếp, có huấn luyện | Train PraNet để tạo segmentation mask |
| Module 2 | Trực tiếp, có fine-tuning | Cải tiến quá trình học nhằm giảm False Positive |
| Module 3 | Sử dụng pretrained encoder | Trích xuất embedding để đối chiếu bằng chứng |
| Module 4 | Gián tiếp | Kết hợp kết quả từ DL bằng classifier nhẹ |

## **7\. Kết luận**

Deep Learning là thành phần cốt lõi của đề tài, đặc biệt trong hai giai đoạn: phân đoạn polyp và cải tiến mô hình bằng Hard-Negative Learning.

Module 3 tiếp tục khai thác đặc trưng do Deep Learning tạo ra để kiểm tra các vùng nghi vấn. Module 4 kết hợp những thông tin này bằng phương pháp Machine Learning nhẹ nhằm đưa ra quyết định cuối cùng.

Điểm khác biệt của đề tài không nằm ở việc sử dụng nhiều mạng neural lớn, mà ở việc nghiên cứu cách huấn luyện và kiểm chứng dự đoán để giảm False Positive trong khi duy trì chất lượng phân đoạn.

**Đóng góp nghiên cứu dự kiến:** Kết hợp cải tiến ở giai đoạn huấn luyện (Hard-Negative Learning) với kiểm chứng candidate ở giai đoạn suy luận (Dual-Evidence Verification). Hiệu quả của hướng kết hợp này cần được chứng minh qua thực nghiệm, ablation study và đánh giá trên dữ liệu ngoài miền.

**MODULE 1  |  POLYP SEGMENTATION BACKBONE**

# **1\. Mục tiêu và phạm vi**

Huấn luyện baseline PraNet để dự đoán polyp probability map và binary mask; lưu mọi đầu ra đủ thông tin để M2 tiếp tục fine-tune và M3/M4 đánh giá. M1 không tự thực hiện DER hoặc loại candidate bằng retrieval.

# **A. Quy định dữ liệu chung**

Tất cả thành viên dùng chung một manifest và một bản chia bệnh nhân. Không được tự chia ảnh lại trong notebook riêng hoặc thay nhãn split cho tiện thực nghiệm.

| Tên split | Vai trò |
| :---- | :---- |
| SEG\_TRAIN | Train PraNet và M2; hard-negative mining chỉ trong dữ liệu được phép huấn luyện. |
| META\_FIT | Tạo candidate và memory; fit reranker bằng dự đoán của model không học trên chính các bệnh nhân này. |
| CALIBRATION | Chọn threshold hoặc cấu hình nhẹ, không dùng để train backbone/memory bank. |
| OUTER\_TEST | Test cuối từng fold; cấm dùng để train, chọn hyperparameter hay xây memory. |
| NEVER\_POLYP\_TEST | Cohort ảnh normal giữ kín để đo FPR cuối; không dùng để mining/tuning. |

   
*Mỗi output phải truy xuất được image\_id, patient\_id, fold, split, run\_id và checkpoint\_id (khi có). Với candidate, bổ sung candidate\_id; với memory, bổ sung bank\_version. OOF chỉ dùng khi checkpoint không từng học trên bệnh nhân tương ứng; dự đoán trên ảnh đã train phải ghi là in\_sample.*

# **B. Input phải có**

| Dữ liệu | Định dạng / thông tin | Nguồn |
| :---- | :---- | :---- |
| Ảnh nội soi | RGB .jpg/.png, lưu đường dẫn ảnh gốc và H×W | FAR |
| Ground-truth mask | .png: 0 \= nền, 255 \= polyp; normal tương ứng mask rỗng | FAR / sinh mask rỗng |
| Dataset manifest | .csv chứa image\_id, patient\_id, is\_polyp, original\_path, mask\_path | Nhóm thống nhất |
| Patient splits | .csv chứa fold/split theo bệnh nhân | Nhóm thống nhất |
| Training config | .yaml: lr, batch, epochs, seed, loss, preprocessing | Người A |

 

# **C. Việc phải làm theo thứ tự**

| ID | Cách thực hiện | Kiểm tra xong |
| :---- | :---- | :---- |
| A01 | Quét toàn bộ dataset; đối chiếu ảnh-mask; đếm ảnh polyp/normal; kiểm tra trùng ID và patient ID. | dataset\_manifest.csv \+ data\_check.json |
| A02 | Áp dụng patient\_splits.csv; viết Dataset class và augmentation chỉ cho SEG\_TRAIN. Resize hình học ảnh-mask đồng bộ. | Notebook đọc dữ liệu không lỗi, 10 cặp ảnh-mask kiểm đúng |
| A03 | Train PraNet baseline fold0 bằng AMP; lưu seed, config, train/val loss và best checkpoint từ validation. | best\_model.pth tải lại và inference được |
| A04 | Dự đoán trên các split cho phép; lưu probability map float32, mask nhị phân và metadata. | H×W đúng ảnh gốc; đủ hàng trong predictions.csv |
| A05 | Tính Dice/IoU/Recall trên ảnh polyp và FPR frame-level trên ảnh normal; báo tách split. | metrics.json kèm threshold và quy tắc vùng ≥100 px |
| A06 | Lưu ít nhất 12 ảnh ví dụ: TP, FP, FN, normal đúng. Xuất hướng dẫn tải output Kaggle. | comparison/ \+ README\_module1.md |

 

# **D. Danh mục OUTPUT cần nộp**

| Tên file/thư mục | Lưu thông tin gì? | Bắt buộc |
| :---- | :---- | :---- |
| checkpoints/best\_model.pth | Trọng số tốt nhất theo validation; architecture version | Có |
| checkpoints/last\_checkpoint.pth | Epoch, model, optimizer/scheduler state để resume | Có |
| predictions/probability/{image\_id}.npy | float32, \[0,1\], H×W gốc, xác suất trước threshold | Có |
| predictions/binary\_mask/{image\_id}.png | Mask 0 hoặc 255; H×W gốc | Có |
| predictions/predictions.csv | Mapping từng ảnh đến checkpoint/config, split, output path | Có |
| metrics/metrics.json | Dice, IoU, pixel Recall, normal frame-FPR và số mẫu | Có |
| logs/history.csv | Epoch, train loss, val loss, val metrics, lr | Có |
| logs/train.log | Lệnh chạy, epoch/time, cảnh báo/lỗi | Có |
| configs/train\_config.yaml | Toàn bộ thông số và threshold | Có |
| visualizations/comparison/ | Ảnh gốc, GT, probability heatmap, binary prediction | Có |
| README\_module1.md | Hướng dẫn khởi tạo môi trường, load checkpoint và inference | Có |

 

# **E. Các cột phải lưu trong predictions.csv**

| Cột | Ý nghĩa |
| :---- | :---- |
| image\_id, patient\_id | Định danh ảnh và bệnh nhân từ manifest |
| fold, split, prediction\_type | Ví dụ: fold=0, META\_FIT, holdout; in\_sample ≠ OOF |
| run\_id, checkpoint\_id | Kết nối output với phiên bản model cụ thể |
| original\_height, original\_width | Kích thước ảnh RGB gốc |
| model\_input\_height, model\_input\_width | Độ phân giải feed vào PraNet |
| probability\_path, mask\_path | Đường dẫn file thật |
| threshold, preprocessing\_version | Ngưỡng tạo mask và version tiền xử lý |
| saved\_at / kaggle\_version | Nguồn phiên bản output có thể tải lại |

 

# **F. Cấu trúc thư mục chuẩn**

| results/ └── module1/     └── fold0/         ├── checkpoints/         │   ├── best\_model.pth         │   └── last\_checkpoint.pth         ├── predictions/         │   ├── probability/         │   ├── binary\_mask/         │   └── predictions.csv         ├── metrics/metrics.json         ├── configs/train\_config.yaml         ├── logs/         │   ├── train.log         │   └── history.csv         ├── visualizations/comparison/         └── README\_module1.md |
| :---- |

 

# **G. Nghiệm thu, lỗi thường gặp và chuyển giao**

· Mở checkpoint trên notebook Kaggle mới và dự đoán lại tối thiểu 3 ảnh; kết quả mask/threshold thống nhất.

· Probability map phải là xác suất thực (không phải binary mask đổi sang float); resize về ảnh gốc đúng thứ tự nội suy.

· Không dùng OUTER\_TEST hoặc NEVER\_POLYP\_TEST để chọn checkpoint/ngưỡng. Ghi rõ FPR cấp ảnh theo vùng liên thông ≥100 pixel tại resolution đã khai báo.

· Ảnh thuộc cùng bệnh nhân không được đi vào đồng thời tập model train và split holdout dùng cho đánh giá/memory.

· Không gọi in\_sample prediction là OOF; chỉ tạo OOF khi checkpoint chưa từng học bệnh nhân đó.

· Bàn giao B: best\_model.pth \+ train\_config.yaml \+ data protocol; bàn giao C/D: predictions.csv \+ probability/masks \+ metric protocol.

# **H. Mẫu báo cáo bàn giao**

| Mục | Cần điền |
| :---- | :---- |
| Task / Run ID | Mã task và run; ví dụ M1-E01-fold0 |
| Cấu hình | Checkpoint, seed, fold, config YAML, Git commit / Kaggle Version |
| Đã làm gì? | Mô tả phương pháp và thay đổi so với baseline |
| Đầu vào | Dataset version, patient split version và nguồn file |
| Đầu ra | Các đường dẫn thực tế đến checkpoints, probability, masks, predictions.csv và metrics.json |
| Số liệu | Metric thật, kèm split, threshold và quy tắc tính |
| Minh họa | Ít nhất ảnh thành công, ảnh thất bại và lý giải |
| Hạn chế | Lỗi chưa xử lý, trade-off Recall-FPR, kế hoạch khắc phục |
| Kiểm tra lại | Notebook load lại output và chạy inference/evaluation được |

   
Xác nhận bàn giao: Người phụ trách \_\_\_\_\_\_\_\_\_\_  |  Người nghiệm thu \_\_\_\_\_\_\_\_\_\_  |  Ngày \_\_\_\_\_\_\_\_\_\_

**MODULE 2  |  HARD-NEGATIVE-AWARE LEARNING**

# **1\. Mục tiêu và phạm vi**

Giảm báo nhầm ngay trong quá trình huấn luyện PraNet bằng cách khai thác ảnh/vùng negative mà model đang dự đoán sai. Module 2 tạo một checkpoint segmentation cải tiến, KHÔNG phải một module xử lý ảnh đứng sau PraNet khi suy luận.

# **A. Quy định dữ liệu chung**

Tất cả thành viên dùng chung một manifest và một bản chia bệnh nhân. Không được tự chia ảnh lại trong notebook riêng hoặc thay nhãn split cho tiện thực nghiệm.

| Tên split | Vai trò |
| :---- | :---- |
| SEG\_TRAIN | Train PraNet và M2; hard-negative mining chỉ trong dữ liệu được phép huấn luyện. |
| META\_FIT | Tạo candidate và memory; fit reranker bằng dự đoán của model không học trên chính các bệnh nhân này. |
| CALIBRATION | Chọn threshold hoặc cấu hình nhẹ, không dùng để train backbone/memory bank. |
| OUTER\_TEST | Test cuối từng fold; cấm dùng để train, chọn hyperparameter hay xây memory. |
| NEVER\_POLYP\_TEST | Cohort ảnh normal giữ kín để đo FPR cuối; không dùng để mining/tuning. |

   
*Mỗi output phải truy xuất được image\_id, patient\_id, fold, split, run\_id và checkpoint\_id (khi có). Với candidate, bổ sung candidate\_id; với memory, bổ sung bank\_version. OOF chỉ dùng khi checkpoint không từng học trên bệnh nhân tương ứng; dự đoán trên ảnh đã train phải ghi là in\_sample.*

# **B. Input phải có**

| Dữ liệu | Dùng để làm gì? | Nguồn |
| :---- | :---- | :---- |
| M1 best\_model.pth \+ train\_config.yaml | Khởi tạo fine-tuning, đảm bảo cùng preprocessing và architecture | M1 |
| SEG\_TRAIN image/mask | Ảnh có polyp, normal và nhãn để học | Dataset chung |
| patient\_splits.csv | Chỉ mine/train đúng các bệnh nhân được phép | Chung |
| M1 predictions / inference script | Chạy mining, tính confidence vùng false positive | M1 |
| Evaluation protocol | So cùng ngưỡng/resolution và Recall | M4/chung |

 

# **C. Công việc chi tiết và đối chứng**

| ID | Cách thực hiện | Bằng chứng phải nộp |
| :---- | :---- | :---- |
| B01 | Load checkpoint M1, tái lập kết quả mẫu; không tự thay split hoặc preprocessing. | sanity\_check.json |
| B02 | Chạy model trên ảnh normal trong SEG\_TRAIN, tạo các connected components bị nhầm; lưu diện tích/confidence/ROI. | hard\_negative\_pool.csv \+ ảnh crop |
| B03 | Train đối chứng 1: PraNet baseline giữ nguyên; đối chứng 2: PraNet \+ normal images được lấy mẫu cân bằng. | checkpoints \+ configs \+ summary |
| B04 | Thử hard-negative re-sampling; tránh chọn toàn easy background. Báo cáo ratio positive/negative. | mining\_config.yaml \+ sampling log |
| B05 | Fine-tune với Dice/BCE và thêm FP-weighted loss khả vi; curriculum bật/tăng trọng số loss theo epoch. | finetune\_config.yaml \+ history.csv |
| B06 | Chạy predictions trên META\_FIT/CALIBRATION/OUTER\_TEST theo protocol; xuất đúng schema M1. | predictions.csv \+ .npy \+ .png |
| B07 | So FPR (cùng Recall) và Dice/IoU; phân tích false-positive giảm được và true polyp bị xóa. | comparison\_metrics.csv \+ failure\_cases/ |

 

# **D. Output bắt buộc**

| Tên file/thư mục | Lưu thông tin gì? | Bắt buộc |
| :---- | :---- | :---- |
| hard\_negative\_pool.csv | Danh sách ảnh/vùng normal bị dự đoán sai và score | Có |
| hard\_negative\_rois/ | Ảnh crop của hard-negative candidates để QA | Có |
| configs/mining\_config.yaml | Ngưỡng chọn mẫu, min area, tần suất mining, top-K | Có |
| configs/finetune\_config.yaml | Optimizer, seed, loss, lambda(t), pos/neg sampling | Có |
| checkpoints/best\_fp\_aware.pth | Trọng số PraNet cải tiến, chọn theo validation | Có |
| checkpoints/last\_checkpoint.pth | Trạng thái resume đầy đủ | Có |
| predictions/probability/ | Probabilities float32 H×W gốc sau fine-tune | Có |
| predictions/binary\_mask/ | Mask sau fine-tune, 0 hoặc 255 | Có |
| predictions/predictions.csv | Metadata tương thích chính xác với M1 | Có |
| metrics/comparison\_metrics.csv | Baseline, normal-trained, hard-negative: Dice/IoU/Recall/FPR | Có |
| metrics/experiment\_summary.json | Tổng hợp đánh giá, config, sample counts, conclusions | Có |
| logs/history.csv | Train/val loss, val Dice/Recall/FPR qua epoch | Có |
| failure\_cases/ | Ảnh FP trước/sau và case làm giảm Recall | Có |
| README\_module2.md | Cách mine, fine-tune, load lại model, export outputs | Có |

 

# **E. Schema hard\_negative\_pool.csv**

| Trường dữ liệu | Ý nghĩa |
| :---- | :---- |
| candidate\_id | ID vùng dự đoán sai; ổn định theo run và image\_id |
| image\_id, patient\_id, fold, split | Nguồn mẫu; split phải được phép train |
| source\_run\_id, source\_checkpoint | Model nào sinh hard-negative candidate |
| x1, y1, x2, y2 | Bounding box tọa độ trên ảnh gốc |
| area\_pixels | Số pixel của connected component sau preprocessing quy ước |
| mean\_confidence, max\_confidence | Xác suất trên vùng bị nhầm |
| mining\_round, sampling\_weight, selected | Lần mining, trọng số chọn mẫu và cờ chọn |
| roi\_path | Ảnh RGB crop được dùng QA/training nếu áp dụng |

 

# **F. Cấu trúc thư mục chuẩn**

| results/ └── module2/     └── fold0/         ├── hard\_negative\_pool.csv         ├── hard\_negative\_rois/         ├── checkpoints/         │   ├── best\_fp\_aware.pth         │   └── last\_checkpoint.pth         ├── configs/         │   ├── mining\_config.yaml         │   └── finetune\_config.yaml         ├── predictions/         │   ├── probability/         │   ├── binary\_mask/         │   └── predictions.csv         ├── metrics/         │   ├── comparison\_metrics.csv         │   └── experiment\_summary.json         ├── failure\_cases/         ├── logs/history.csv         └── README\_module2.md |
| :---- |

 

# **G. Nghiệm thu và lưu ý kỹ thuật**

· Chạy được tối thiểu 3 biến thể: baseline, thêm normal thông thường, thêm hard-negative (giữ cùng data split và metric).

· Hard-negative mining chỉ trên SEG\_TRAIN. Các checkpoint mining phải có provenance; không lén xem kết quả OUTER\_TEST hoặc NEVER\_POLYP\_TEST khi chọn mẫu.

· Chuyển hard-negative vùng sang trọng số cho BCE hoặc loss khả vi trên probability; KHÔNG kỳ vọng gradient qua threshold/connected components.

· Báo FPR giảm kèm Recall/Dice thay đổi; phải chọn/cố định operating point từ validation/calibration.

· Khi dữ liệu có nhiều ảnh normal, cân bằng sampling để tránh model dự đoán toàn background.

· M3 phải nhận predictions cùng schema M1 và liên kết được source\_checkpoint với đúng memory bank; nếu đổi checkpoint thì build lại memory.

· Có ít nhất ảnh FP được xóa đúng, ảnh polyp giữ đúng, và ví dụ polyp thật bị mất sau fine-tune.

# **H. Mẫu báo cáo bàn giao**

| Mục | Cần điền |
| :---- | :---- |
| Task / Run ID | Mã task và run; ví dụ M2-E01-fold0 |
| Cấu hình | Checkpoint, seed, fold, config YAML, Git commit / Kaggle Version |
| Đã làm gì? | Mô tả phương pháp và thay đổi so với baseline |
| Đầu vào | Dataset version, patient split version và nguồn file |
| Đầu ra | Các đường dẫn thực tế đến checkpoint, hard\_negative\_pool.csv, predictions.csv và comparison\_metrics.csv |
| Số liệu | Metric thật, kèm split, threshold và quy tắc tính |
| Minh họa | Ít nhất ảnh thành công, ảnh thất bại và lý giải |
| Hạn chế | Lỗi chưa xử lý, trade-off Recall-FPR, kế hoạch khắc phục |
| Kiểm tra lại | Notebook load lại output và chạy inference/evaluation được |

   
Xác nhận bàn giao: Người phụ trách \_\_\_\_\_\_\_\_\_\_  |  Người nghiệm thu \_\_\_\_\_\_\_\_\_\_  |  Ngày \_\_\_\_\_\_\_\_\_\_

**MODULE 3  |  DUAL-EVIDENCE CANDIDATE VERIFICATION**

# **1\. Mục tiêu và phạm vi**

Từ probability map/mask của segmentation model, tách từng vùng nghi ngờ, trích feature và đo mức giống nhau với polyp thật (positive memory) hoặc lỗi FP từng gặp (hard-negative memory). Output là bằng chứng định lượng cho M4; không được tự coi similarity cao là nhãn cuối cùng.

# **A. Quy định dữ liệu chung**

Tất cả thành viên dùng chung một manifest và một bản chia bệnh nhân. Không được tự chia ảnh lại trong notebook riêng hoặc thay nhãn split cho tiện thực nghiệm.

| Tên split | Vai trò |
| :---- | :---- |
| SEG\_TRAIN | Train PraNet và M2; hard-negative mining chỉ trong dữ liệu được phép huấn luyện. |
| META\_FIT | Tạo candidate và memory; fit reranker bằng dự đoán của model không học trên chính các bệnh nhân này. |
| CALIBRATION | Chọn threshold hoặc cấu hình nhẹ, không dùng để train backbone/memory bank. |
| OUTER\_TEST | Test cuối từng fold; cấm dùng để train, chọn hyperparameter hay xây memory. |
| NEVER\_POLYP\_TEST | Cohort ảnh normal giữ kín để đo FPR cuối; không dùng để mining/tuning. |

   
*Mỗi output phải truy xuất được image\_id, patient\_id, fold, split, run\_id và checkpoint\_id (khi có). Với candidate, bổ sung candidate\_id; với memory, bổ sung bank\_version. OOF chỉ dùng khi checkpoint không từng học trên bệnh nhân tương ứng; dự đoán trên ảnh đã train phải ghi là in\_sample.*

# **B. Input phải có**

| Input | Định dạng | Lưu ý |
| :---- | :---- | :---- |
| Ảnh nội soi gốc | RGB .jpg/.png | Crop ROI từ ảnh RGB, không crop từ mask |
| Probability map \+ binary mask | .npy float32 \+ .png 0/255 | Cùng kích thước ảnh gốc |
| predictions.csv | CSV của M1 hoặc M2 | Biết model/checkpoint tạo candidate |
| Ground-truth mask | .png | Chỉ sử dụng GT khi tạo nhãn mẫu TRAIN/META\_FIT; không dùng GT lúc test |
| patient\_splits.csv | .csv | Memory hoàn toàn không nhận ảnh test |
| Encoder pretrained | Backbone encoder / frozen DINOv2-S | Ghi version/weights/preprocessing |

 

# **C. Các bước làm cụ thể**

| ID | Làm như thế nào? | Đầu ra kiểm tra |
| :---- | :---- | :---- |
| C01 | Threshold probability (theo config), tách connected components thành candidate riêng, tạo unique candidate\_id. | candidates.csv \+ candidate\_masks/ |
| C02 | Crop local ROI và ROI mở rộng có context, giới hạn bbox trong ảnh; lưu cả crop coords gốc. | roi\_metadata.csv \+ ROI .png |
| C03 | Chạy frozen encoder; trích embedding float32, map row index ↔ candidate\_id chính xác. | embeddings.npy \+ embedding\_index.csv |
| C04 | Gán nhãn trên META\_FIT bằng GT: positive/negative/ambiguous theo IoU hoặc overlap threshold rõ ràng. | labels trong candidates.csv |
| C05 | Xây positive\_bank và hard\_negative\_bank từ candidate hợp lệ; lưu patient\_id, checkpoint, quality, nguồn ảnh. | banks .npy \+ bank\_metadata.csv |
| C06 | Top-K cosine retrieval từ hai banks; có leave-one-patient-out khi fit scores trên META\_FIT. | retrieval\_scores.csv \+ topk\_neighbors.csv |
| C07 | Train ROI classifier đơn giản trên cùng embedding làm đối chứng; so positive-only / negative-only / dual. | roi\_classifier.pkl \+ ablation.csv |

 

# **D. Output bắt buộc**

| File/thư mục | Thông tin lưu | Bắt buộc |
| :---- | :---- | :---- |
| candidates/candidates.csv | Tọa độ, confidence, candidate\_id, nhãn GT nếu được phép | Có |
| candidates/candidate\_masks/ | Mask từng vùng (đúng kích thước ảnh gốc) | Có |
| roi/roi\_local/, roi\_context/ | Ảnh cắt hai scale | Có |
| roi/roi\_metadata.csv | Crop coords, padding, ROI path và source model | Có |
| embeddings/embeddings.npy | Feature matrix dạng N×d, float32 | Có |
| embeddings/embedding\_index.csv | Row-to-candidate mapping, encoder version | Có |
| memory/positive\_bank.npy | Vector cho true-polyp candidates | Có |
| memory/negative\_bank.npy | Vector cho hard-FP candidates | Có |
| memory/bank\_metadata.csv | Mỗi vector có patient, source, quality và bank\_version | Có |
| retrieval/retrieval\_scores.csv | Top-K mean/std/similarity margin | Có |
| retrieval/topk\_neighbors.csv | Mỗi query-neighbor một hàng để dễ audit | Có |
| baselines/roi\_classifier.pkl | Classifier dùng chung embeddings, không memory | Có |
| metrics/ablation.csv | Positive-only, negative-only, dual, ROI classifier | Có |
| configs/retrieval\_config.yaml | Crop ratio, encoder ID, K, metric, label policy | Có |
| retrieval/retrieval\_examples/ | Ảnh query \+ top positive/negative neighbors | Có |
| README\_module3.md | Cách xây bank, query, và tái lập score | Có |

 

# **E. Trường thông tin phải lưu**

| Tập tin | Các trường không được thiếu |
| :---- | :---- |
| candidates.csv | candidate\_id; image\_id; patient\_id; fold; split; source\_run\_id; bbox x1,y1,x2,y2; area\_pixels; mean\_confidence; max\_confidence; candidate\_mask\_path; gt\_label (chỉ train/fit). |
| roi\_metadata.csv | candidate\_id; local\_bbox; context\_bbox; context\_ratio; image\_height/width; roi\_local\_path; roi\_context\_path; resize\_transform. |
| embedding\_index.csv | embedding\_row; candidate\_id; encoder\_id; embedding\_dim; roi\_version; embedding\_norm. |
| bank\_metadata.csv | bank\_version; bank\_type; embedding\_row; candidate\_id; patient\_id; source\_checkpoint; quality\_flag; labeling\_rule. |
| retrieval\_scores.csv | candidate\_id; source\_run\_id; bank\_version; encoder\_id; positive\_similarity; negative\_similarity; similarity\_margin; positive\_std; negative\_std; topk\_positive\_ids; topk\_negative\_ids. |
| topk\_neighbors.csv | candidate\_id; bank\_type; neighbor\_id; neighbor\_patient\_id; rank; cosine\_similarity; bank\_version. |

 

# **F. Cấu trúc thư mục chuẩn**

| results/ └── module3/     └── fold0/         ├── candidates/         │   ├── candidates.csv         │   └── candidate\_masks/         ├── roi/         │   ├── roi\_local/         │   ├── roi\_context/         │   └── roi\_metadata.csv         ├── embeddings/         │   ├── embeddings.npy         │   └── embedding\_index.csv         ├── memory/         │   ├── positive\_bank.npy         │   ├── negative\_bank.npy         │   └── bank\_metadata.csv         ├── retrieval/         │   ├── retrieval\_scores.csv         │   ├── topk\_neighbors.csv         │   └── retrieval\_examples/         ├── baselines/roi\_classifier.pkl         ├── configs/retrieval\_config.yaml         ├── metrics/ablation.csv         └── README\_module3.md |
| :---- |

 

# **G. Tiêu chí nghiệm thu và những lỗi phải tránh**

· Bounding box và ROI kiểm tra trực quan đúng tọa độ ảnh gốc; candidate\_mask không phải binary mask của toàn ảnh.

· Positive và Negative Memory sử dụng cùng pipeline model-prediction → candidate → ROI → encoder, không lấy positive GT crops theo một cách khác.

· Không đánh đồng vùng overlap thấp với negative khi GT polyp thật có thể nằm trong candidate; có lớp ambiguous/ignore nếu cần.

· Memory phải có provenance và không chứa ảnh OUTER\_TEST/NEVER\_POLYP\_TEST; khi fit retriever/reranker trên META\_FIT dùng retrieval loại trừ cùng patient.

· Khi thay baseline checkpoint bằng M2 checkpoint, tạo lại candidate/embeddings/bank tương ứng và tăng version; không ghi đè bank cũ.

· Đối chứng ROI classifier phải dùng cùng encoder, cùng candidate, cùng split để kết luận retrieval có tác dụng thật hay không.

· Chạy được cả baseline M1 và checkpoint M2; có ví dụ top-K hợp lý và cases retrieval thất bại.

# **H. Mẫu báo cáo bàn giao**

| Mục | Cần điền |
| :---- | :---- |
| Task / Run ID | Mã task và run; ví dụ M3-E01-fold0 |
| Cấu hình | Checkpoint, seed, fold, config YAML, Git commit / Kaggle Version |
| Đã làm gì? | Mô tả phương pháp và thay đổi so với baseline |
| Đầu vào | Dataset version, patient split version và nguồn file |
| Đầu ra | Các đường dẫn thực tế đến candidates, ROI, embeddings, banks, retrieval scores và ROI classifier |
| Số liệu | Metric thật, kèm split, threshold và quy tắc tính |
| Minh họa | Ít nhất ảnh thành công, ảnh thất bại và lý giải |
| Hạn chế | Lỗi chưa xử lý, trade-off Recall-FPR, kế hoạch khắc phục |
| Kiểm tra lại | Notebook load lại output và chạy inference/evaluation được |

   
Xác nhận bàn giao: Người phụ trách \_\_\_\_\_\_\_\_\_\_  |  Người nghiệm thu \_\_\_\_\_\_\_\_\_\_  |  Ngày \_\_\_\_\_\_\_\_\_\_

**MODULE 4  |  RELIABILITY-AWARE DECISION & EVALUATION**

# **1\. Mục tiêu và phạm vi**

Ghép confidence và bằng chứng truy xuất của M3, đưa ra quyết định keep/reject từng candidate để tạo final segmentation mask. Đồng thời xây evaluation dùng chung cho M1–M4, ablation, kiểm soát FPR/Recall, phân tích sai và demo. Không dùng GT để ra quyết định trên ảnh test hoặc ảnh đưa vào demo.

# **A. Quy định dữ liệu chung**

Tất cả thành viên dùng chung một manifest và một bản chia bệnh nhân. Không được tự chia ảnh lại trong notebook riêng hoặc thay nhãn split cho tiện thực nghiệm.

| Tên split | Vai trò |
| :---- | :---- |
| SEG\_TRAIN | Train PraNet và M2; hard-negative mining chỉ trong dữ liệu được phép huấn luyện. |
| META\_FIT | Tạo candidate và memory; fit reranker bằng dự đoán của model không học trên chính các bệnh nhân này. |
| CALIBRATION | Chọn threshold hoặc cấu hình nhẹ, không dùng để train backbone/memory bank. |
| OUTER\_TEST | Test cuối từng fold; cấm dùng để train, chọn hyperparameter hay xây memory. |
| NEVER\_POLYP\_TEST | Cohort ảnh normal giữ kín để đo FPR cuối; không dùng để mining/tuning. |

   
*Mỗi output phải truy xuất được image\_id, patient\_id, fold, split, run\_id và checkpoint\_id (khi có). Với candidate, bổ sung candidate\_id; với memory, bổ sung bank\_version. OOF chỉ dùng khi checkpoint không từng học trên bệnh nhân tương ứng; dự đoán trên ảnh đã train phải ghi là in\_sample.*

# **B. Input phải có**

| Dữ liệu | Vai trò | Từ đâu |
| :---- | :---- | :---- |
| candidates.csv \+ candidate\_masks/ | Vùng phải ra quyết định và mask để tái tạo ảnh cuối | M3 |
| retrieval\_scores.csv | positive/negative similarity, agreement, margin | M3 |
| predictions.csv \+ probability maps | Gốc segment và confidence, image mapping | M1/M2 |
| GT masks \+ patient splits | Gán nhãn meta-fit, đo Dice/IoU/FPR độc lập | Dataset |
| calibration subset | Chọn threshold/operating point theo protocol | Protocol chung |

 

# **C. Công việc cần làm**

| ID | Làm như thế nào? | Bằng chứng bàn giao |
| :---- | :---- | :---- |
| D01 | Viết hàm evaluation độc lập từ sớm: pixel Dice, IoU, Recall; frame FPR đúng min-area, per-image/per-patient. | test\_metrics.py \+ evaluation\_config.yaml |
| D02 | Join candidates và retrieval\_scores theo candidate\_id; kiểm tra uniqueness, missing fields và nguồn model/bank. | feature\_schema.json \+ fit\_features.csv |
| D03 | Train Logistic Regression với features confidence, s+, s-, margin và agreement; scaler nếu có phải lưu. | reranker.pkl \+ training\_config.yaml |
| D04 | Tạo retrieval-agreement/reliability feature và so với baseline confidence-only, margin-only và ROI classifier. | ablation\_results.csv |
| D05 | Chọn threshold bằng CALIBRATION. Không dùng OUTER\_TEST hoặc NEVER\_POLYP\_TEST để tinh chỉnh. | threshold\_selection.csv |
| D06 | Keep/reject từng candidate, ghép thành final mask cùng H×W gốc; xuất candidate\_decisions.csv. | final\_masks/ \+ final\_predictions.csv |
| D07 | Đánh giá segmentation/FPR/Recall trên held-out; so matched Recall hoặc trình bày Pareto trade-off. | metrics\_summary.json \+ per patient/image metrics |
| D08 | Phân tích FP được loại đúng, FP còn sót, polyp thật bị loại; chạy external set nếu dữ liệu/split phù hợp. | failure\_analysis.csv \+ visualization \+ demo |

 

# **D. Output bắt buộc**

| File/thư mục | Thông tin lưu | Bắt buộc |
| :---- | :---- | :---- |
| reranker/reranker.pkl | Logistic Regression và scaler nếu fit | Có |
| reranker/feature\_schema.json | Thứ tự tên features, dtype, cách biến đổi | Có |
| reranker/threshold\_selection.csv | Ngưỡng thử và FPR/Recall trên CALIBRATION | Có |
| decisions/candidate\_decisions.csv | Điểm rerank, threshold và keep/reject từng candidate | Có |
| final\_predictions/final\_masks/ | Final binary mask H×W ảnh gốc | Có |
| final\_predictions/final\_predictions.csv | Mapping final mask, checkpoint, bank, reranker | Có |
| evaluation/metrics\_by\_image.csv | Dice/IoU/Recall ảnh polyp; alarm flag ở ảnh normal | Có |
| evaluation/metrics\_by\_patient.csv | Đánh giá/tổng hợp theo bệnh nhân | Có |
| evaluation/metrics\_summary.json | Các metric chính, sample count và operating point | Có |
| evaluation/ablation\_results.csv | Base, \+normal, \+HN, \+dual, \+reliability, ROI classifier | Có |
| evaluation/bootstrap\_ci.csv | 95% CI theo patient (nếu có đủ thời gian) | Khuyến nghị |
| failure\_analysis/failure\_analysis.csv | FP removed, still FP, TP removed, FN cases | Có |
| failure\_analysis/comparison/ | Ảnh GT, baseline mask, final mask, marked mistakes | Có |
| configs/evaluation\_config.yaml | Metric definition, resolution, min area, threshold | Có |
| configs/inference\_config.yaml | IDs của model/bank/reranker, preprocessing | Có |
| external/external\_test.csv | Thử PolypGen nếu khả thi | Mở rộng |
| demo/ | Notebook hoặc script nhập ảnh và xuất final mask | Có |
| README\_module4.md | Cách load bundle và chạy demo từ ảnh mới | Có |

 

# **E. Schema bắt buộc và ví dụ report**

| File | Các trường tối thiểu |
| :---- | :---- |
| candidate\_decisions.csv | candidate\_id; source\_run\_id; bank\_version; reranker\_version; base\_confidence; positive\_similarity; negative\_similarity; reliability\_score; final\_score; decision\_threshold; keep; candidate\_mask\_path. |
| final\_predictions.csv | image\_id; patient\_id; fold; split; checkpoint\_id; bank\_version; reranker\_version; output\_mask\_path; runtime\_ms; inference\_config\_id. |
| metrics\_summary.json | run\_id; fold; evaluation cohort; number of patients/images; Dice; IoU; pixel Recall; FPR\_frame; min\_component\_pixels; threshold; evaluation\_resolution. |
| failure\_analysis.csv | image\_id; patient\_id; candidate\_id; gt\_overlap; base\_decision; final\_decision; error\_type; image\_path; observation. |

 

| {   "run\_id": "E05\_fold0",   "fold": 0,   "segmentation\_checkpoint": "m2\_fold0\_best",   "bank\_version": "dual\_bank\_v1",   "reranker\_version": "lr\_v1",   "decision\_threshold": null,   "metrics": {     "dice": null,     "iou": null,     "pixel\_recall": null,     "never\_polyp\_frame\_fpr": null,     "test\_fold\_normal\_frame\_fpr": null   },   "evaluation": {     "min\_component\_pixels": 100,     "resolution": "record\_in\_config"   } } |
| :---- |

   
*Các giá trị null chỉ là mẫu báo cáo, tuyệt đối không ghi số liệu dự kiến thành kết quả thật.*

# **F. Cấu trúc thư mục chuẩn**

| results/ └── module4/     └── fold0/         ├── reranker/         │   ├── reranker.pkl         │   ├── feature\_schema.json         │   └── threshold\_selection.csv         ├── decisions/candidate\_decisions.csv         ├── final\_predictions/         │   ├── final\_masks/         │   └── final\_predictions.csv         ├── evaluation/         │   ├── metrics\_by\_image.csv         │   ├── metrics\_by\_patient.csv         │   ├── metrics\_summary.json         │   ├── ablation\_results.csv         │   └── bootstrap\_ci.csv         ├── failure\_analysis/         │   ├── failure\_analysis.csv         │   └── comparison/         ├── external/external\_test.csv         ├── configs/         │   ├── evaluation\_config.yaml         │   └── inference\_config.yaml         ├── demo/         └── README\_module4.md |
| :---- |

 

# **G. Nghiệm thu và những lỗi nguy hiểm**

· Join nguồn dữ liệu bằng candidate\_id và version, không join chỉ theo thứ tự dòng vì dễ sai mapping.

· Bảo đảm một candidate bị reject thực sự biến mất khỏi final mask; candidate giữ lại giữ vùng tương ứng. Final H×W phải khớp ảnh.

· Final probability map chỉ là tùy chọn; nếu có suppress theo candidate thì không gọi đó là xác suất đã calibration nếu chưa chứng minh.

· Fit reranker ở META\_FIT với retrieval LOO theo bệnh nhân; chọn threshold trên CALIBRATION; khóa đánh giá cuối trên OUTER\_TEST và NEVER\_POLYP\_TEST.

· So FPR tại Recall tương đương bằng threshold chọn trên calibration; báo Recall pixel-level khác lesion/frame sensitivity.

· FPR frame-level: ảnh normal được báo nhầm nếu tồn tại connected component \>=100 pixel, tính trên resolution đã ghi rõ. Không dùng Precision để thay FPR.

· Với các ảnh liên tiếp của cùng bệnh nhân, CI/bootstrap nên lấy mẫu theo bệnh nhân thay vì coi các frame độc lập.

· Reranker loại candidate không thể tự phát hiện polyp bị base model bỏ sót hoàn toàn; nêu rõ trong failure analysis.

· Demo nhận ảnh mới, tự load đúng checkpoint/bank/reranker, chạy được không dùng GT hoặc sửa code tay.

# **H. Mẫu báo cáo bàn giao**

| Mục | Cần điền |
| :---- | :---- |
| Task / Run ID | Mã task và run; ví dụ M4-E01-fold0 |
| Cấu hình | Checkpoint, seed, fold, config YAML, Git commit / Kaggle Version |
| Đã làm gì? | Mô tả phương pháp và thay đổi so với baseline |
| Đầu vào | Dataset version, patient split version và nguồn file |
| Đầu ra | Các đường dẫn thực tế đến reranker, final masks, metrics, ablation, failure analysis và demo |
| Số liệu | Metric thật, kèm split, threshold và quy tắc tính |
| Minh họa | Ít nhất ảnh thành công, ảnh thất bại và lý giải |
| Hạn chế | Lỗi chưa xử lý, trade-off Recall-FPR, kế hoạch khắc phục |
| Kiểm tra lại | Notebook load lại output và chạy inference/evaluation được |

   
Xác nhận bàn giao: Người phụ trách \_\_\_\_\_\_\_\_\_\_  |  Người nghiệm thu \_\_\_\_\_\_\_\_\_\_  |  Ngày \_\_\_\_\_\_\_\_\_\_

[image1]: <data:image/png;base64,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admyZTVNvjWW22KLLXo+VVjAr1y5svz6179GZCayyNSrcO+8W2RGewnP5blz5/a1tykxTGTqUzntnmuzSJ90aN7KV3z0ox+tAj23BaNDJ0WmgoNeeWnHI4rM/Lr8+OOPr6959Ir8s5/9bH29s9lmm9U2vHupHRG9MpJzzUEKAN56moRJXosqc/nll9cArVfgStMfrJxzzjl9r8v1alJBSq/a5SeiyMyvy/WgqQfQTVVk6lgiU77y2GOPrel6W3T11VeX22+/vdpn7733rrb91re+VW0rvymbzZ8/H5GZsMj86U9/Wr73ve/V74I1P5VnkSnBKXtrPvt1udJVT2UUs3K7mwLDXpdrrStdtlM81xuPv/3bv63xXJ8a5LZgdOikyNRE89NhFJn6+F8iM364rqBy8cUX1wX85S9/ue6Aqg2la/L6w+KvfOUrfYENAN56moRJXov6oxStU61XfQeoNO1kqIzS9Ed/rrN8+fLyvve9r6Z/7nOfq390YZGpeieffHJta//9968+Q2mbssjUdestkES5hLf8px7YJdxlH9lJtvz6179eba4Arz/I0pskROZEZA9/hqDPuuJfk8c//FFskp1V7sUXX6zzW3bV+aB/IWHciSJT6I/yZA+t3fhZjNa01q/su77/MgK89XRGZAIAAABAd0BkAgAAAEDrIDIBAAAAoHUQmQAAAADQOohMAAAAAGgdRCYAAAAAtA4iEwAAAABaB5EJAAAAAK3TGZG5atUqAAAAgAlkvQCjQ2dEJgAAAAB0B0QmAAAAALQOIhMAAAAAWgeRCQAAAACtg8gEAAAAgNZBZAIAAABA6yAyAQAAAKB1EJkAAAAA0DqITAAAAABonc6IzNWrV5dp06ZVNt988/L000/3lWliwYIFtc78+fN7adOnTy+LFy/uK2v233//snTp0r50ANj42A/MmTOnlyb/MGvWrF6emDFjRmOa68g36HzFihUT2p/MP3SVN954o1x00UX1+j7ykY/0+dCf/exn5V3vele10y233FLT3nzzzXLdddeVvfbaa0IbKqM2XPeuu+6qabvuumv5xS9+0df3OPDqq6/W2CM+85nPTMjTPMrz7LXXXiuHHHJIL01zUWVPPPHEvvkoe++5555lyy23LGeccUZf311Fcy2vOxNjekzP9nI7Wq86dkyP5LZhdOiUyPz4xz9eA8K1115b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mZD/6U9/uqarnPIkrOLukgRI3nFUmR/84Ad150l897vfnSAOonDIgvHhhx+edDc21zEqu+OOO/bGpraUHkXmG2+8UXfnNC5d1/HHH98bs65P5dT/008/3deviCJT3HXXXdXWEt5qY88996xtvOc97+nZS+W08+Zxve9975sg1GU//Ua7yj761ety7ZZuttlmE8ahPF2Hrvnv/u7van1d00UXXdQ3ZhFFZn5I0K71rrvu2uv7lltuqekeg4j3uWlO5P7uv//+ctZZZ/XOpyoyjz766N71Z5GZ77va14NR7hsAAKALdFJk6ns/Befzzz+/Ly+jcttuu2399Sty88UvfnGCgFRAl+DJIiWjV+rKu/POO+u5hMhURaZ3vtzfQQcd1Cc0m16Xa5yxnF6rSmTfc889E0SmhMtWW21V7rjjjlruD3/4Q3nooYeqOD322GNrmkSTRJ1eWcd+hcWSytx7771l7733LgsXLuy18Zvf/KaW07XLBhKdFmtuQ7t8TTaMdol1Hn300bor/cILL9Rz7fLNnTu3jkFi79prr63pzzzzTLX9s88+2zduXfv73//++imE+hT69lV58b77IeKGG26o53mMg+ZE7u/EE08sTz75ZO883y+3GUWm7p+uRzbTQ49F5oc//OFy+OGH17y7776716bnQe4bAACgC3ROZCqw6w9KtEOXRWMTCuIf+MAHagDX95dOl6DZZ599JggD4V27LD4iSo+vkLOQjGIq50kcaqdMO4Gxv9i+62jXy9+QStxJIH35y18u++23XxXO3n2LIlO7XzqWOLR9Fi1a1HedcYyRLJaE2nEbLucdSI1BglbCVruMFtFNNox9RpEpO8qeFsZnnHFGufHGG2t6HovIu4XCNhAScfrmNeafeeaZVdDvsMMOE8YRxzhsTuT+JBAtioXsJhv4nn3hC1/oG5fQN6q//vWve23EvCxmX3755XLAAQc07qQCAACMOp0SmRIhCtIKyFMRmEJlzz333PpHFBIZ/otuC7lLL720/PCHP+xx8803Nwqk3KYEgs+zkIwiJuapXe1aSThoPHr9O0xkZjGlnVuln3LKKeVHP/pRo8jUbpl22XT+jne8o1x//fW9fAmgeK1PPfVU37VZLEkoSZjlV/KxrEWSPx+wWDr11FMbbRjtknc/9Wpcfej+SERrt1Jt6PtFpcdx+49jIiojNC/U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wj/rrLMmpCtNi3sqTn4yR4DIBGgHrUuzYMGC3voVm2++eXn66adruRkzZvTSDzzwwPLwww9PaGfx4sU178gjj+ylqY7azH1uLBCZa4d8sO+57uWKFSt691n84Ac/KG+++WYt67RtttmmnH/++WXNmjW9duKciu3l/jKvvfZa+fKXv9yXLi644IIaA2666aa+PLOpiEyt0S233LLaVb85f2257LLLyi677FKPbWfZTel53Zu99967nHbaaeU3v/lN+eAHP1h+8pOf9JWB0aZzInOPPfYozzzzTE179dVXy9vf/nZEJsCIoXX005/+tAoIrTvx8Y9/vJ5fe+215bDDDqvrV6Jg/vz5NX233XYrBx988IR2ovhwGiKz28gHey6sXLmyvPHGG/U+a7488sgjZYsttuiJvJ133rncc8895dRTT60PJzfeeGOvHQlRtaF6l19+ea+93F9mWBzQvLLwzXlmUxCZWptao1qrjz/+eLnooov6yqwPtnNOH4Rjs+ZJzoPRplMi86ijjuqhtDPOOKMeW2TuueeedTdE2+8KWFokDzzwQNl+++3r1vwll1xSHZXqKn3HHXcszz77bJk3b159wkJkArSD1pHWk8+1PhWcfb7ddtuVBx98cILI3HXXXcsXv/jFCe0oqKgtvTI75phj6k5WFJnbbrttueGGG8p+++1XfYF2X0488cSa98ILL1QfIeGh9f+73/2ufOc73yl33HFH7W/33Xcvy5Yt6xu72paoVXCVz5DokaA56aST6o6a0tWefIjK67rUnsagclFwKG+zzTardXSd6jv3t6kh28S5IHSfPV/22Wef3j20yDzuuOPqseZMbk/1LD7ivd56662rva+77rpyzTXX1DTthj7xxBNl//33r/dMu5qxLd0j97lw4cJ67zUHlK5jlbHIHBRbxoHXX3+9HH/88fV19SuvvNJL19rQ9XttXHXVVb21cffdd1f0tlFp5513Xrnwwgt7ZS0sX3zxxZ6dtVaUrnuhclqTsr1iuNr2Wv/Zz35W/YAeJrRmvSP61FNPVb+Rxw+jQ6dEpia3nmQlDrWbuddee/UWvPK32mqrctttt9XymtQKZNqW32mnncqjjz5aXn755XLAAQfUfKV7cspJySEhMgHaIb7ClACIIvOxxx6rO5beyXS5733ve3VXK7Zjkek1qlepDjwKfhKeCmg614Ol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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAApkAAADBCAYAAACE7avhAAA6MElEQVR4Xu2d669V1dnF338ACGAgbcAUAxJMpEZKvDQoNmJoolGRxDbeMIY0tTRWrWk/tI0YSaMFLzGRFtvQ+KGKiRArtVqptYoasWirGC8REYJYvIAIiRCarDdjNmN3nGfPtfZtncs+Z3z45aw175dnPnPsudbe5//+85//FMYYY4wxxtTJ/8UAY4wxxhhjesUi0xhjjDHG1I5FpjHGGGOMqR2LTGOMMcYYUzsWmaZWpkyZUixevLgpHKxevbq46aabii+//LIRtnz58mLcuHHF5s2bm9IbY+pj7969xZw5c9LfGNcO27dvr1zf7XLw4MHi2LFjTeEjBfQTfTx06FBTXJ1wPHP1oH7ExXDT33z00UfF9ddf3xTeLlg327ZtawofyXQtMiEMyIQJE4p77rmn+Pjjj5vSmcEDok3n4IwzzsjOwbRp04obbrihKXwwaLUJzZ49u/jnP//ZuM+JzNNOO6244IILmvLWyZo1a4qJEyem6+PHjxfnn39+GsPTTz+92LhxY1N6Y4Ya2OVll11WTJo0Ka2RuXPnNqVZsWJFWt8xPEeZyPze977XVhl1iMwjR44US5YsKe69996mOJSN/sRwtlv3HLSB7WEY1vALL7zQyKPpTz755OL+++9vKjtHP4hM2Madd95ZLF26tDh8+HBTfCd88cUXxV133dXYR95+++2mNKZ9uKdpGOaYtvjaa6815WmXBx54IJWBOWPYrl270rw9/vjjTelHAj2JTAwm73/0ox+lsJdffrkprQIxgXS5hdULWLA5BzqagcjUOYDAxNjCOek84OQQzhufomIZdfHHP/6xePTRR5Nwg1D75JNPUjg+tV133XVpI7vqqqua8uVEJsCHlldffbUpfZ1gcd52223JSdNR/+IXv0jtgROP6Y0ZKvChEHb40EMPNU7+d+zYMSAN1gc+LLWyVdj2k08+Wfzud79LYhJ/9+zZk+JwKtJOGaAOkZmD/gnrMcYBikzUr+FsD30++oExO3DgQApDHPPggyM24liGAp8FHwa/Ch+2YcOGpjTklVdeSWMaw8GiRYuKm2++uSk80ovIxIdwiEz0C2MT4zth9+7djdOxf//732kMN23a1JTOtEeVyIz7XCuijUcw/6eeemqy+QsvvLA4evRoU5rh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>
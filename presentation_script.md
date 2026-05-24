# Bài Thuyết Trình Đồ Án — GSP26AI08
## Multi-Agent System for Crowd Behavior Analysis in Metro Station
### Phần 1: Slide 1–8 | Phần 2: Slide 40–48 | Tổng thời lượng: ~15 phút

---

# PHẦN 1: GIỚI THIỆU ĐỒ ÁN (Slide 1–8)
## ⏱ Thời lượng ước tính: ~8 phút

---

## 🎙 SLIDE 1 — Trang bìa (~1 phút)

> *[Đứng thẳng, nhìn thẳng vào hội đồng, giọng rõ ràng, tự tin]*

Kính thưa quý thầy cô trong hội đồng bảo vệ đồ án,

Em tên là **Hồ Hải Thuận**, sinh viên thuộc nhóm **GSP26AI08**. Hôm nay, em xin được trình bày đồ án tốt nghiệp với đề tài:

**"Multi-Agent System for Crowd Behavior Analysis in Metro Station"** — hay nói cách khác, đây là một **hệ thống đa tác nhân nhân tạo** được thiết kế để phân tích hành vi đám đông trong môi trường nhà ga tàu điện ngầm.

Trong khoảng thời gian hôm nay, em sẽ lần lượt trình bày về: bối cảnh và động lực của đề tài, kiến trúc hệ thống tổng thể, từng mô-đun kỹ thuật, và kết quả thực nghiệm đạt được. Kính mời hội đồng theo dõi.

---

## 🎙 SLIDE 2 — Hệ thống là gì? (~1 phút)

Trước hết, em xin giới thiệu tổng quan về hệ thống mà nhóm đã xây dựng.

Hệ thống của chúng em được xây dựng dựa trên kiến trúc **Multi-Agent System — hệ thống đa tác nhân**. Đây là một mô hình kiến trúc phần mềm trong đó nhiều tác nhân AI độc lập, mỗi tác nhân đảm nhận một vai trò riêng biệt, phối hợp với nhau để tạo ra một hệ thống phân tích tổng thể và toàn diện.

Cụ thể, hệ thống của chúng em gồm ba đặc tính cốt lõi:

- **Autonomous Agents** — Mỗi tác nhân là một đơn vị AI độc lập, có chức năng và nhiệm vụ xác định rõ ràng. Ví dụ: tác nhân phát hiện người, tác nhân ước lượng mật độ, tác nhân nhận diện hành động.
- **Collaborative Analysis** — Các tác nhân không hoạt động đơn lẻ mà phối hợp với nhau theo thời gian thực, tổng hợp thông tin để đưa ra phân tích toàn diện về tình trạng đám đông.
- **Metro Environment** — Toàn bộ hệ thống được tối ưu hóa đặc biệt cho môi trường nhà ga tàu điện ngầm — nơi có mật độ người cực cao, điều kiện ánh sáng phức tạp, và yêu cầu phản ứng theo thời gian thực.

---

## 🎙 SLIDE 3 — Tại sao cần MAS cho nhà ga tàu điện ngầm? (~1.5 phút)

Kính thưa hội đồng, để làm rõ hơn về động lực của đề tài, em xin trình bày những thách thức thực tiễn mà bài toán này đặt ra.

**Thứ nhất**, các hệ thống camera an ninh truyền thống hiện nay có những hạn chế cơ bản: chúng không được thiết kế để xử lý đám đông mật độ cao, phụ thuộc nhiều vào con người để giám sát, và thường phản ứng chậm — nghĩa là chỉ xử lý được sau khi sự cố đã xảy ra, chứ không thể phòng ngừa chủ động.

**Thứ hai**, nhà ga tàu điện ngầm là môi trường đặc biệt nguy hiểm vì mật độ người quá cao. Sự chen lấn, ngã ngã, hay các tình huống khẩn cấp có thể xảy ra rất nhanh và lan rộng trong vài giây.

**Thứ ba**, đây là lý do cốt lõi mà nhóm chọn hướng tiếp cận MAS: khả năng **phát hiện chủ động theo thời gian thực**. Thay vì chờ con người phát hiện sự cố, hệ thống của chúng em tự động nhận diện, theo dõi và phân loại hành vi của từng cá nhân trong đám đông, phát cảnh báo sớm trước khi tình huống nguy hiểm leo thang.

---

## 🎙 SLIDE 4 — Ứng dụng thực tiễn (~1 phút)

Về mặt ứng dụng thực tiễn, hệ thống của chúng em mang lại giá trị trên ba khía cạnh:

**An toàn và bảo mật**: Hệ thống giám sát liên tục 24/7, phát hiện ngay lập tức các tình huống nguy hiểm như té ngã hay hành vi bất thường, giúp lực lượng an ninh phản ứng nhanh hơn đáng kể so với phương pháp truyền thống.

**Hiệu quả vận hành**: Bằng cách tối ưu hóa luồng người và phân bổ nhân lực an ninh dựa trên dữ liệu mật độ thực tế, hệ thống giúp nhà ga vận hành hiệu quả hơn, giảm thời gian chờ và ùn tắc.

**Tiết kiệm chi phí**: Việc tự động hóa giám sát giúp giảm chi phí nhân lực dài hạn, đồng thời giảm thiểu tổn thất từ các sự cố tai nạn có thể phòng ngừa được.

---

## 🎙 SLIDE 5–6 — Các phương pháp liên quan và hạn chế (~1.5 phút)

Kính thưa hội đồng, để định vị hướng tiếp cận của chúng em trong bức tranh nghiên cứu hiện tại, em xin điểm qua ba hướng kỹ thuật chính trong lĩnh vực phân tích đám đông:

**Hướng thứ nhất — Ước lượng mật độ (Density Estimation)**: Các mô hình như MCNN và CSRNet nhận ảnh đầu vào và tạo ra density map — bản đồ mật độ người. Ưu điểm là xử lý tốt khi số lượng người rất đông và có nhiều che khuất. Tuy nhiên, hạn chế cơ bản là chúng không thể đặc tả được **trạng thái và hành vi của từng cá nhân** trong đám đông.

**Hướng thứ hai — Phát hiện và theo dõi (Detection & Tracking)**: YOLO và Faster R-CNN cho độ chính xác cao trong việc phát hiện người. Các thuật toán theo dõi như SORT và ByteTrack duy trì được danh tính của từng người qua thời gian. Tuy vậy, hiệu suất suy giảm đáng kể trong điều kiện đám đông dày đặc do vấn đề che khuất lẫn nhau.

**Hướng thứ ba — Phân loại hành động (Action Classification)**: Các mô hình như ST-GCN và AGCN nhận đầu vào là các điểm khung xương và phân loại chúng theo nhãn hành động. Đây là hướng cho phép phân tích chi tiết ở cấp độ từng cá nhân, phát hiện các tình huống nguy hiểm như té ngã.

**Vấn đề cốt lõi** mà nhóm chúng em nhận thấy: **mỗi hướng đơn lẻ đều có điểm yếu riêng**. Chỉ ước lượng mật độ thì không biết ai đang làm gì. Chỉ phát hiện đối tượng thì mất hiệu quả khi đám đông dày. Chỉ phân loại hành động thì thiếu bức tranh tổng thể. Đây chính là lý do chúng em đề xuất kiến trúc MAS — **kết hợp cả ba năng lực** trong một hệ thống thống nhất, mỗi tác nhân bổ sung cho nhau.

---

## 🎙 SLIDE 7 — Kiến trúc prototype (~0.5 phút)

> *[Chỉ vào sơ đồ trên slide]*

Đây là kiến trúc tổng thể của hệ thống prototype mà nhóm đã xây dựng. Luồng xử lý đi từ trái sang phải: video đầu vào từ camera → **tác nhân phát hiện người** xác định vị trí từng cá nhân → **tác nhân ước lượng tư thế** trích xuất 17 điểm khung xương COCO → **tác nhân phân loại hành động** nhận diện walking, standing, sitting, falling → **lớp tổng hợp** đưa ra cảnh báo tổng thể.

---

## 🎙 SLIDE 8 — Dataset tự thu thập (~1 phút)

Về dữ liệu, một điểm đặc biệt của đồ án này là nhóm đã **tự thu thập dataset thực tế** tại môi trường nhà ga tàu điện ngầm, thay vì chỉ sử dụng các dataset công khai có sẵn.

Dataset tự thu thập bao gồm **19 video** với tổng thời lượng **13 phút 5 giây**, được ghi hình tại các khu vực đặc trưng: **7 cổng soát vé, 4 khu vực cầu thang và 8 sân ga**. Độ dài mỗi video dao động từ 9 giây đến 1 phút 13 giây, phản ánh đa dạng các tình huống thực tế.

Việc tự thu thập dữ liệu mang lại hai lợi thế quan trọng: (1) đảm bảo tính đặc thù với môi trường nhà ga Việt Nam — điều mà các dataset nước ngoài không có; (2) cho phép chúng em kiểm soát hoàn toàn quy trình gán nhãn theo yêu cầu của bài toán.

---
---

# PHẦN 2: MÔ HÌNH PHÂN LOẠI HÀNH ĐỘNG — ST-GCN (Slide 40–48)
## ⏱ Thời lượng ước tính: ~7 phút

---

## 🎙 SLIDE 40 — Giới thiệu mô-đun Action Classification (~0.5 phút)

Kính thưa hội đồng, em xin chuyển sang phần thứ hai của bài trình bày — **mô-đun phân loại hành động**, đây là tác nhân cốt lõi và cũng là phần đóng góp kỹ thuật chính của em trong đồ án.

Nhiệm vụ của mô-đun này là: nhận đầu vào là chuỗi khung xương của một người qua nhiều frame, và đưa ra phán đoán người đó đang thực hiện hành động nào trong 4 lớp: **đứng, đi, ngồi, hay ngã**.

---

## 🎙 SLIDE 41 — Bài toán phân loại hành động (~1 phút)

> *[Chỉ vào chuỗi frame trên slide]*

Đây là minh họa cụ thể của bài toán. Hệ thống nhận một chuỗi các frame liên tiếp — ở đây là 5 frame — mỗi frame chứa thông tin khung xương của một người. Dựa trên **mẫu chuyển động qua thời gian** của các khớp xương, mô hình đưa ra kết luận: **Class: Walking**.

Điều quan trọng ở đây là bài toán **không chỉ nhìn vào một frame duy nhất** — một người đứng yên và một người bắt đầu té ngã có thể trông rất giống nhau trong một frame. Chính vì vậy, cần khai thác thông tin **cả không gian lẫn thời gian** — đây là nền tảng của kiến trúc ST-GCN mà em sẽ trình bày ngay sau đây.

---

## 🎙 SLIDE 42 — Dataset Le2i Fall (~1 phút)

Về dữ liệu huấn luyện cho mô-đun này, ngoài dataset tự gán nhãn, chúng em còn sử dụng bổ sung **Le2i Fall Dataset** — một dataset công khai được cộng đồng nghiên cứu về phát hiện té ngã sử dụng rộng rãi.

Dataset này gồm **48 video** với tổng **14.668 frame**. Điểm đáng chú ý: **9.256 frame** chứa nhãn `fall_label=1` — tức là trạng thái bình thường, và **5.412 frame** chứa nhãn té ngã thực sự. Đây cho thấy **sự mất cân bằng lớp** rõ rệt — một thách thức kỹ thuật quan trọng mà chúng em phải giải quyết trong quá trình huấn luyện. Thống kê bounding box cho thấy chiều cao trung bình khoảng 126 pixel và chiều rộng 78 pixel — đây là kích thước khá nhỏ, phản ánh độ khó của việc trích xuất keypoint chính xác.

---

## 🎙 SLIDE 43–44 — Dataset tự gán nhãn & Tiền xử lý (~1 phút)

Song song với Le2i, chúng em cũng **tự gán nhãn** dữ liệu từ các video tự thu thập. Quá trình gán nhãn sử dụng công cụ CVAT, mỗi track người được đánh nhãn thủ công với một trong 4 lớp hành động.

Sau khi có dữ liệu thô, chúng em áp dụng một pipeline tiền xử lý gồm nhiều bước: **nội suy keypoint** để lấp đầy các điểm bị mất do che khuất, **imputation kinematic-spatial** để ước tính vị trí khớp dựa trên cấu trúc sinh học, và **chuẩn hóa tọa độ** theo trung tâm khung hình.

Về augmentation, nhóm áp dụng: lật ngang (flip LR), thêm nhiễu Gaussian, xoay ngẫu nhiên, và scale jitter — nhằm tăng tính đa dạng của dữ liệu huấn luyện và cải thiện khả năng tổng quát hóa của mô hình. Sau augmentation, tập train đạt khoảng **180 track** với tỉ lệ cân bằng **45 mẫu mỗi lớp**.

---

## 🎙 SLIDE 45 — Mô hình ST-GCN (~1.5 phút)

Kính thưa hội đồng, em xin trình bày mô hình chính — **ST-GCN (Spatial-Temporal Graph Convolutional Network)**, được chúng em chọn làm **mô hình benchmark** cơ sở để so sánh hiệu năng.

ST-GCN là mô hình tiên phong trong lĩnh vực nhận diện hành động dựa trên dữ liệu khung xương, được Yan et al. đề xuất tại AAAI 2018. Điểm đột phá của mô hình là: **áp dụng mạng tích chập đồ thị đồng thời trên cả hai chiều không gian và thời gian**.

Về không gian, cơ thể người được biểu diễn như một đồ thị: các khớp xương là node, các kết nối xương là cạnh. Phép tích chập đồ thị cho phép mô hình học được **mối tương quan giữa các khớp liền kề** — ví dụ như mối liên hệ giữa chuyển động của vai và khuỷu tay khi vung tay.

Về thời gian, tích chập temporal kernel kích thước `9×1` cho phép mô hình nắm bắt **xu hướng chuyển động qua 9 frame liên tiếp** — đủ để phân biệt giữa người đang đứng yên và người đang bắt đầu té ngã.

Đầu vào của mô hình là tensor dạng `(N, C=3, T=100, V=17)` — tương ứng với N mẫu, 3 channels tọa độ x/y/score, 100 frames, và 17 khớp xương theo chuẩn COCO. Mô hình đi qua **10 ST-GCN block** với kênh tăng dần: 64→64→64→64→128→128→128→256→256→256. Sau đó qua Global Average Pooling và một lớp Fully Connected để cho ra xác suất của 4 lớp hành động.

Để tận dụng kiến thức học sẵn, chúng em nạp **pretrained weights từ MMAction2** — bộ trọng số được huấn luyện trên NTU-RGB+D 60, sau đó fine-tune trên dataset của dự án.

---

## 🎙 SLIDE 46 — Các mô hình khác: SGCN và CTR-GCN (~1.5 phút)

Ngoài ST-GCN, nhóm còn nghiên cứu và cài đặt hai kiến trúc bổ sung để so sánh:

**SGCN — Sparse Graph Convolutional Network**: Đây là một biến thể nhẹ của GCN, sử dụng pointwise group convolution kết hợp với channel shuffle mechanism nhằm **giảm tải tính toán**. Ưu điểm của SGCN là tiêu thụ ít tài nguyên phần cứng, phù hợp cho triển khai trực tiếp tại nhà ga — nơi hệ thống cần chạy liên tục trên hardware nhúng. Đây là mô hình được xem xét cho deployment thực tế.

**CTR-GCN — Channel-wise Topology Refinement Graph Convolution Network**: Đây là kiến trúc cao cấp hơn. Điểm khác biệt cốt lõi so với ST-GCN là: thay vì dùng **một đồ thị topology duy nhất** cho tất cả các channel, CTR-GCN có khả năng **tự học và tinh chỉnh topology riêng cho từng channel**. Điều này cho phép mô hình nắm bắt các mối tương quan chuyển động phức tạp hơn, đặc biệt hữu ích khi phân biệt các hành động có hình dáng khung xương tương tự nhau như "đứng" và "bắt đầu ngã". CTR-GCN cho độ chính xác cao nhất trong số ba mô hình so sánh.

Với ba mô hình này, chúng em có thể thực hiện **so sánh đa chiều**: giữa accuracy và computational cost, giữa độ phức tạp mô hình và khả năng generalization — từ đó đưa ra khuyến nghị cho từng kịch bản triển khai khác nhau.

---

## 🎙 SLIDE 47–48 — Kết quả thực nghiệm (~0.5 phút)

> *[Chỉ vào bảng kết quả / confusion matrix trên slide nếu có]*

Về kết quả thực nghiệm, ba mô hình được đánh giá trên cùng một tập validation không qua augmentation, đảm bảo tính công bằng trong so sánh. Các chỉ số chính bao gồm: Accuracy tổng thể, Macro F1-score, và đặc biệt là **Fall Recall** — chỉ số phản ánh khả năng phát hiện đúng các trường hợp té ngã, lớp thiểu số và quan trọng nhất trong bài toán an toàn.

Kết quả cho thấy ST-GCN — với việc tận dụng pretrained weights — đạt được nền tảng tốt và ổn định. CTR-GCN vượt trội về accuracy nhờ kiến trúc tinh tế hơn. SGCN thể hiện sự cân đối tốt giữa tốc độ và hiệu năng, phù hợp cho triển khai thực tế.

---

## 🎙 Kết luận phần Action Classification (~0.5 phút)

Tóm lại, trong mô-đun phân loại hành động, chúng em đã:
- Xây dựng và gán nhãn dataset thực tế tại nhà ga
- Thiết kế pipeline tiền xử lý và augmentation chuyên biệt
- Cài đặt và so sánh ba kiến trúc GCN: ST-GCN, SGCN, và CTR-GCN
- Đánh giá toàn diện trên các chỉ số phù hợp với bài toán an toàn

Đây là mô-đun đóng góp trực tiếp vào khả năng phát hiện sự cố của toàn hệ thống MAS.

Kính thưa hội đồng, em xin kết thúc phần trình bày của mình tại đây. Em rất mong nhận được câu hỏi và góp ý quý báu từ quý thầy cô. Xin trân trọng cảm ơn!

---

## 📝 Ghi chú phân bổ thời gian

| Phần | Slide | Thời gian |
|---|---|---|
| Trang bìa & giới thiệu | 1 | ~1 phút |
| Hệ thống là gì? | 2 | ~1 phút |
| Tại sao cần MAS? | 3 | ~1.5 phút |
| Ứng dụng thực tiễn | 4 | ~1 phút |
| Phương pháp liên quan | 5-6 | ~1.5 phút |
| Kiến trúc prototype | 7 | ~0.5 phút |
| Dataset tự thu thập | 8 | ~1 phút |
| **Tổng Phần 1** | **1-8** | **~8 phút** |
| Giới thiệu Action Classification | 40 | ~0.5 phút |
| Bài toán & minh họa | 41 | ~1 phút |
| Dataset Le2i | 42 | ~1 phút |
| Dataset tự gán nhãn & preprocessing | 43-44 | ~1 phút |
| Mô hình ST-GCN | 45 | ~1.5 phút |
| SGCN & CTR-GCN | 46 | ~1.5 phút |
| Kết quả & kết luận | 47-48 | ~0.5 phút |
| **Tổng Phần 2** | **40-48** | **~7 phút** |
| **Tổng cộng** | | **~15 phút** |

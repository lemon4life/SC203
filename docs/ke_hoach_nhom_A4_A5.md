# Kế hoạch nhóm cho Assignment 4 và Assignment 5

**Nhóm:** Team 1 (LEMMEPASS), SC203 Scientific Methods, lớp 25A02
**Cập nhật:** Chủ nhật 08/11/2026
**Đề tài:** Topic 3, mô hình DoSSR (làm nét ảnh), kiểm tra xem chữ trên ảnh có được phục hồi nhất quán khi đổi seed không
**Repo:** https://github.com/lemon4life/SC203 (nhánh `feat/baseline-runner`)

> Các giờ giấc trong file này là **đề xuất**. Nếu thấy không hợp lịch thì báo trong nhóm để đổi, nhưng **giữ giờ nộp** ở mục 1.
> Chỗ nào ghi **TODO** là chưa biết chắc. Đừng đoán, hãy hỏi.

---

## 1. Hạn nộp

| Bài | Hạn | Yêu cầu |
|---|---|---|
| **Group Assignment 4: SOTA's Limitations** | **07:00 thứ Ba 10/11/2026** | Chạy thí nghiệm để tìm bằng chứng và phân tích giới hạn của phương pháp hiện có liên quan đến đề tài |
| **Group Assignment 5: Baseline** | **07:00 thứ Ba 17/11/2026** | Báo cáo về baseline (cách làm, kết quả thí nghiệm, ...) |

- Mỗi bài nộp **một file PDF**, tên file: `<StudentID1>_<StudentID2>_....pdf` (ghi mã số sinh viên của các thành viên. TODO: xác nhận có cần ghi cả 4 mã số không).
- **Giờ nộp nội bộ cho Assignment 4: 22:00 thứ Hai 09/11** (để còn dự phòng, đừng nộp lúc 06:50 sáng thứ Ba).
- Assignment 5: bản nháp **thứ Sáu 13/11**, nộp tối **thứ Hai 16/11**.

---

## 2. Bức tranh chung (nói đơn giản)

Mô hình DoSSR biến ảnh mờ thành ảnh nét. Một vấn đề hay gặp là đôi khi nó **viết sai chữ** trên biển báo, và kết quả **đổi khi mình đổi seed**. Câu hỏi của nhóm:

> Nếu thêm một luật nhỏ lúc dạy mô hình (LoRA), nó có viết đúng chữ nhất quán hơn khi đổi seed không?

Cho **Assignment 4**, mình chưa cần dạy LoRA. Mình chỉ cần **chứng minh bằng thí nghiệm thật** rằng giới hạn này có tồn tại: chạy mô hình với 10 seed, rồi chấm điểm phần chữ.

> **Lưu ý quan trọng:** hiện nhóm **chưa đo lỗi chữ nào**. "DoSSR viết chữ không nhất quán" mới chỉ là giả thuyết. Chỉ viết kết luận sau khi có điểm số. Nếu kết quả cho thấy mô hình khá ổn định, thì báo cáo đúng như vậy.

---

## 3. Các từ hay dùng

| Từ | Hiểu đơn giản |
|---|---|
| **Seed** | Một lần tung xúc xắc. Cùng một lần tung thì ra cùng một bức ảnh. Mình dùng chung 10 lần tung cho mọi ảnh để bài kiểm tra công bằng. |
| **Prompt** | Câu ghi chú gửi cho mô hình. "Mặc định" là ghi chú có sẵn (`clean, high-resolution, 8k`), "rỗng" là không ghi chú. |
| **Ảnh LR / HR** | LR là ảnh mờ đầu vào (128x128). HR là ảnh nét gốc (512x512), dùng làm đáp án. |
| **Hộp chữ + bản chép** | Một khung chữ nhật vẽ quanh mỗi biển chữ, cùng với chữ trên biển đó gõ ra. Đây là đáp án cho phần chữ. |
| **OCR** | Một con robot đọc chữ trong ảnh. Nó đọc ảnh đầu ra của mô hình, rồi mình so với đáp án. |
| **Điểm chính** | Như bài kiểm tra chính tả: biển nào sai dù **chỉ một** chữ cái thì cả biển bị tính là sai. |
| **CER** | Tỉ lệ chữ cái sai trên tổng số chữ cái. |
| **Bất đồng giữa các seed** | 10 bạn học sinh trả lời cùng một câu hỏi. Nếu mỗi bạn một đáp án khác nhau thì mô hình chưa ổn định. |
| **LoRA** | Một bản vá nhỏ để huấn luyện, thay vì dạy lại cả mô hình khổng lồ (như dán thêm một miếng vá nhỏ thay vì xây lại cả ngôi nhà). |

---

## 4. Đã có gì rồi (có thể đưa vào báo cáo)

Nguồn chi tiết: `docs/after_01_dossr_baseline_discovery.md` và `context.md`. Chỉ dùng số liệu có trong các file đó.

**Cách làm và cài đặt**
- Dữ liệu: `RealSRVal_crop128`, gồm `test_LR` (100 ảnh 128x128) và `test_HR` (100 ảnh 512x512), tên file giống nhau từng cặp.
- DoSSR chạy trên Colab với GPU NVIDIA A100 80GB. Cài đặt: 5 bước, scale x4 (128 lên 512), fp16, seed 42, color fix wavelet.
- Nhóm viết bộ chạy (runner) đọc file cấu hình, ghi log, lưu kết quả theo từng seed.

**Kiểm tra bản vá bộ nhớ**
- Bản vá attention (SDPA) cho kết quả gần như giống code gốc: khác nhau trung bình 0,03 mức xám (tối đa 2) trên ảnh thật, và bộ nhớ như nhau ở ảnh 512.
- Kiểm tra wrapper prompt: chế độ mặc định cho ảnh giống hệt khi không có wrapper (sai khác 0); chế độ rỗng cho ảnh khác.

**Tốc độ và bộ nhớ trên đầu vào thật (128 lên 512)**
- Bộ nhớ đỉnh khoảng 7,32 GiB, khoảng 1,75 giây mỗi ảnh (bản vá bật). Đo trên một ảnh.

**Seed**
- Cùng seed chạy hai lần cho ảnh giống hệt nhau (sai khác 0,0).
- Khác seed (42 so với 43) làm pixel đổi nhiều: trung bình 8,82 mức xám, tối đa 255 (trên 3 ảnh).

**Prompt**
- Prompt mặc định là `clean, high-resolution, 8k`. Lúc huấn luyện, DoSSR dùng prompt rỗng.
- Trên ảnh Canon_001, đổi prompt mặc định sang rỗng làm pixel đổi trung bình 0,37 mức xám, nhỏ hơn nhiều so với đổi seed (5,17 trên cùng ảnh). Đây mới là **gợi ý từ một ảnh**, cần kết quả 10 seed để kết luận.

**Giới hạn bộ nhớ khi ảnh lớn (thí nghiệm ngoài ý muốn nhưng có giá trị)**
- Lúc đầu nhóm lỡ dùng ảnh `test_HR` làm đầu vào (512 lên 2048). Lần chạy đầu, chưa vá, đòi 80 GiB bộ nhớ và hết bộ nhớ GPU.
- Sau khi vá attention, đỉnh bộ nhớ là 36,04 GiB (fp16). Sau khi vá thêm VAE, còn 20,51 GiB.
- Khi viết, ghi rõ đây là thí nghiệm với đầu vào lớn bất ngờ, và nêu bài học về chọn đúng thư mục `test_LR`.

**Đã chạy (cần kiểm tra lại cho đủ)**
- Chạy 10 seed, cả prompt mặc định và prompt rỗng, trên 100 ảnh.
- Hình eduroam (Canon_001) so sánh LR, HR và hai đầu ra.
- Điểm PSNR/SSIM/LPIPS đã chạy thử trên 5 ảnh với 5 seed (chưa xem số liệu).

---

## 5. Còn thiếu gì và mức quan trọng

### Assignment 4 (hạn 10/11)

| Việc | Mức độ |
|---|---|
| Kiểm tra 10 seed đã đủ (2.000 ảnh, đúng kích thước 512x512) | **Bắt buộc** |
| Đếm ảnh có chữ đọc được trong `test_HR`, vẽ hộp và chép chữ | **Bắt buộc** |
| Chọn OCR và kiểm tra OCR trên ảnh nét | **Bắt buộc** |
| Chấm điểm chữ (tỉ lệ biển sai, CER, bất đồng giữa các seed) | **Bắt buộc** |
| Hình eduroam | **Bắt buộc** |
| Viết và ghép file PDF | **Bắt buộc** |
| PSNR, SSIM, LPIPS qua 10 seed | Nên có |
| So sánh prompt mặc định và rỗng | Nên có |
| Đoạn về giới hạn bộ nhớ (đã có số liệu) | Nên có |
| Một trang CCSR, PNFS, TADiSR (lấy từ bài báo, ghi rõ là tài liệu) | Nên có |
| Real-CE x4, LoRA, instruction | Chưa cần |

### Assignment 5 (hạn 17/11)

| Việc | Mức độ |
|---|---|
| Mô tả cách làm, cài đặt chính xác (commit, cấu hình) | **Bắt buộc** |
| Bảng 10 seed: PSNR, SSIM, LPIPS (trung bình ± độ lệch chuẩn), thời gian, bộ nhớ | **Bắt buộc** |
| Kiểm tra tính đúng đắn và giới hạn | **Bắt buộc** |
| Điểm chữ và hình eduroam (dùng lại từ Assignment 4) | Nên có |
| Bảng prompt mặc định và rỗng | Nên có |
| Bảng bật/tắt bản vá SDPA | Tùy chọn |
| Chạy thử LoRA kèm VRAM | Tùy chọn (thuộc báo cáo của mentor) |

---

## 6. Phân công và hạn

| Thành viên | Việc chính | Hạn |
|---|---|---|
| **TV1 Hoàng Nguyên Anh** (chạy GPU, ghép bài) | Kiểm tra 2.000 ảnh. Gửi đường dẫn kết quả cho cả nhóm. Gộp 2 file CSV nhãn. Thu mã số sinh viên. Ghép và nộp PDF. | Kiểm tra và gửi đường dẫn **CN tối**. Gộp CSV **CN 21:30**. Nộp **T2 22:00** |
| **TV2 Huỳnh Trần Gia Hân** (đáp án chữ và chấm điểm) | Đánh dấu ảnh 1 đến 50. Báo số lượng ảnh có chữ. Chấm điểm chữ trên các seed. | Nhãn và số đếm **CN 21:00**. Điểm **T2 15:00** |
| **TV3 Hồ Quang Minh** (kiểm tra OCR, LoRA tạm dừng) | Đánh dấu ảnh 51 đến 100. Kiểm tra OCR trên ảnh nét. Giúp sửa script chấm điểm. | Nhãn **CN 21:00**. Chọn OCR **CN tối**. Kiểm tra **T2 12:00** |
| **TV4 Nguyễn Phan Minh Nhật** (báo cáo và hình) | Khung LaTeX. Viết các mục có sẵn dữ liệu. Chạy PSNR/SSIM/LPIPS. Một trang bài báo liên quan. | Khung và mục setup **CN 21:00**. Một trang **T2 12:00**. Metrics **T2 15:00** |
| **Cả nhóm** | Chốt dữ liệu, cùng viết kết quả và phân tích, đọc soát. | Chốt **T2 18:00**. Đọc soát **T2 21:00** |

LoRA (TV3) quay lại từ **thứ Ba 11/11**. Hỏi mentor hạn của lần chạy thử LoRA. **TODO.**

---

## 7. Ai chờ ai

```
Kết quả 10 seed của TV1  ->  TV2 (chấm điểm chữ), TV3 (kiểm tra), TV4 (metrics, hình)
Nhãn của TV2 + TV3       ->  TV1 gộp thành một file CSV  ->  TV2 chấm điểm
Điểm chữ của TV2         ->  TV4 (phần kết quả trong báo cáo)
```

Kết quả được lưu theo từng seed (seed_42 xong trước), nên TV2 và TV3 có thể thử OCR sớm trên seed_42 mà không cần chờ cả 10 seed.

---

## 8. Hướng dẫn chi tiết

### Chuẩn bị chung (15 phút, cho TV2, TV3, TV4)

1. Clone repo và chuyển nhánh:
   ```bash
   git clone https://github.com/lemon4life/SC203
   cd SC203
   git checkout feat/baseline-runner
   ```
2. Đọc ghi chú của mình: `docs/handoff_member2.md`, `docs/handoff_member3.md` hoặc `docs/handoff_member4.md`.
3. Cần Python (phiên bản ghi trong ghi chú) và quyền vào thư mục Drive chung `lemmepass/`. Máy Windows có thể phải gõ `py` thay cho `python`.
4. **Không cần commit git.** Cứ lưu file vào thư mục Drive chung hoặc gửi cho TV1.

---

### 8.1. Thành viên 1: chạy GPU và ghép bài

1. **Kiểm tra 2.000 ảnh** của hai lần chạy 10 seed (mặc định và rỗng). Mỗi seed phải có 100 ảnh PNG, đúng 512x512:
   ```python
   from pathlib import Path
   from PIL import Image

   root = Path("[ĐƯỜNG DẪN mặc định]")   # làm lại cho cả thư mục prompt rỗng
   for seed_dir in sorted(root.glob("seed_*")):
       pngs = sorted(seed_dir.glob("*.png"))
       sai = [p.name for p in pngs if Image.open(p).size != (512, 512)]
       print(seed_dir.name, len(pngs), "ảnh, sai kích thước:", sai[:3])
   ```
2. **Gửi cho nhóm:** hai đường dẫn kết quả (`[ĐƯỜNG DẪN mặc định]`, `[ĐƯỜNG DẪN rỗng]`) và giờ chốt. Chia sẻ quyền xem thư mục Drive cho 3 bạn.
3. **Push code** (`git push origin feat/baseline-runner`) để mọi người clone được script và ghi chú.
4. **Gộp CSV:** khi TV2 và TV3 gửi `data/text_boxes_realsr_m2.csv` (ảnh 1 đến 50) và `data/text_boxes_realsr_m3.csv` (ảnh 51 đến 100), gộp thành `data/text_boxes_realsr.csv`. Script gộp và kiểm tra (`scripts/merge_text_boxes.py`) đang nhờ agent làm. **TODO:** chưa có. Nó phải báo lỗi nếu thiếu ảnh, trùng ảnh, hoặc còn ô `readable` để trống. Gửi file đã gộp cho TV2.
5. **Thu mã số sinh viên** để đặt tên file PDF. Xem lại hình eduroam để chắc khung cắt có chứa chữ.
6. **Ghép bài, đọc soát, nộp** trước 22:00 thứ Hai. Nộp xong thì chụp màn hình xác nhận.
7. **Chỉ TV1 hoặc agent sửa `context.md`.** Mọi người báo tiến độ trong chat, TV1 cập nhật.

Nhờ agent (nếu chưa làm): script gộp CSV, so lệnh trong tin nhắn với tham số thật bằng `--help`, thêm dòng tham chiếu OCR trên ảnh nét (trần) và ảnh phóng bicubic (sàn) vào `text_eval.py`.

---

### 8.2. Thành viên 2: đáp án chữ và chấm điểm

**Mục tiêu:** vẽ khung quanh mỗi chữ đọc được trong ảnh nét và ghi chữ đó là gì.

1. **Tạo contact sheet** (mỗi tờ gồm 20 ảnh nhỏ, có thước ở viền):
   ```bash
   python scripts/make_text_contact_sheets.py --hr-dir "<đường dẫn test_HR>" --out-dir "outputs/text_contact_sheets" --csv-out "data/text_boxes_realsr_m2.csv"
   ```
   Chạy đúng sẽ in: `DONE | total_images=100 | sheets_written=5`.
2. **Đánh dấu ảnh 1 đến 50** (tờ 1, tờ 2 và nửa đầu tờ 3, tức ảnh 41 đến 50). Với mỗi chữ đọc được (biển, tên hãng, số), ghi một dòng trong CSV:
   ```csv
   image,box_id,x,y,w,h,transcript,readable
   ```
   - `x`, `y`: góc trên trái của khung. `w`, `h`: chiều rộng và cao. Tính bằng **pixel của ảnh 512x512** (thước ghi 128, 256, 384).
   - `transcript`: đúng như bạn đọc, **không đoán**.
   - `readable`: `yes` nếu người đọc được chữ trong ảnh nét.
   - Ảnh không có chữ đọc được: ghi theo hướng dẫn trong `docs/handoff_member2.md`.
   - Hãy nghĩ nó như đáp án bài chính tả. Không chắc một chữ thì làm theo ghi chú và hỏi trong nhóm. Dùng đúng luật trong ghi chú (một khung là một từ hay một dòng, chữ hoa chữ thường, khoảng trắng, dấu câu), vì TV3 làm cùng luật.
3. **Tối nay:** báo nhóm trong 50 ảnh có bao nhiêu ảnh có chữ đọc được, và gửi TV1 file CSV.
4. **Chấm điểm chữ** khi có kết quả 10 seed và file CSV đã gộp:
   ```bash
   pip install rapidocr-onnxruntime
   python scripts/text_eval.py --study-root "[ĐƯỜNG DẪN mặc định]" --compare-study-root "[ĐƯỜNG DẪN rỗng]" --boxes-csv "data/text_boxes_realsr.csv" --ocr-backend rapidocr
   ```
   Chạy thử trên **seed_42** trước, rồi mới chạy cả 10 seed. Gửi TV1 thư mục kết quả.
5. Script cho ra: tỉ lệ biển có ít nhất một chữ sai, CER, mức bất đồng giữa các seed. Điểm được gộp theo **từng ảnh** trước (10 seed của một ảnh không độc lập với nhau), và so sánh cặp theo cặp ảnh-seed.

---

### 8.3. Thành viên 3: kiểm tra OCR (LoRA tạm dừng đến thứ Ba)

**A. Đánh dấu ảnh 51 đến 100**, cùng cách với TV2 (bước 1 đến 2 ở mục 8.2) nhưng lưu vào file riêng:

```bash
python scripts/make_text_contact_sheets.py --hr-dir "<đường dẫn test_HR>" --out-dir "outputs/text_contact_sheets" --csv-out "data/text_boxes_realsr_m3.csv"
```

Tờ 3 dùng chung: TV3 lấy ảnh 51 đến 60, TV2 lấy 41 đến 50. Tối nay báo nhóm số ảnh có chữ đọc được trong 50 ảnh của bạn, và gửi TV1 file CSV.

**B. Kiểm tra robot OCR trên ảnh NÉT.** Việc này rất quan trọng: nếu robot không đọc được chữ trên ảnh nét thì điểm số trên ảnh đầu ra của mô hình không có ý nghĩa.

1. Cài `pip install rapidocr-onnxruntime`. Nếu được, thử thêm `easyocr` và so sánh.
2. Với mỗi khung đã đánh dấu, cắt ảnh nét (HR) và cho OCR đọc. Đếm xem nó đọc **đúng hoàn toàn** bao nhiêu khung so với bản chép (cùng luật như trong ghi chú).
3. Thử phóng mỗi ảnh cắt lên 2 lần và 4 lần trước khi đọc. Cách nào tốt hơn?
4. Báo nhóm: dùng công cụ nào, phóng mấy lần, kết quả (ví dụ: đúng 31 trên 40). **Làm xong trong tối nay** để nhóm chọn công cụ OCR.
5. Nếu `scripts/text_eval.py` đã có tùy chọn cho ảnh nét thì dùng. Nếu chưa, làm tay và báo TV1.

**C. Sáng thứ Hai:** chạy `scripts/text_eval.py` chỉ trên seed_42 (câu lệnh ở mục 8.2), gửi TV1 mọi lỗi kèm toàn bộ nội dung lỗi, và giúp TV2 sửa lỗi.

---

### 8.4. Thành viên 4: báo cáo, hình và số liệu

1. **Overleaf:** dùng cùng mẫu với bài nộp đề tài (`topic-3.tex`).
2. **Khung báo cáo Assignment 4** (xem mục 9). Viết ngay các mục 2, 3.1 (phần đầu) và 3.5 từ `docs/after_01_dossr_baseline_discovery.md`.
3. **Luật khi viết:**
   - Chỉ dùng số liệu có trong log hoặc tài liệu của nhóm.
   - Ghi rõ "ước tính" cho số ước tính.
   - Ghi "theo bài báo" cho thông tin về phương pháp khác.
   - **Không** viết kết luận về lỗi chữ trước khi có điểm số.
4. **Hình eduroam** đã làm xong, nằm trên Drive: `outputs/figures/canon_001_eduroam_default_vs_empty_seed42.png`. Đưa vào báo cáo (TODO: TV1 xác nhận khung cắt đúng).
5. **Metrics (PSNR, SSIM, LPIPS):** chạy `scripts/compute_metrics.py` với `--device cpu` trên các thư mục 10 seed (mặc định và rỗng). Câu lệnh chính xác nằm trong `docs/handoff_member4.md`. Gửi TV1 các file CSV.
6. **Một trang CCSR, PNFS, TADiSR:** điền bảng trong `docs/related_methods_ccsr_pnfs_tadisr.md`. Với mỗi bài: vấn đề nó giải quyết, nó xử lý độ ổn định hoặc chữ thế nào, nó cần gì, giới hạn chính (theo lời tác giả), điểm khác của mình. Chỉ dùng thông tin từ bài báo, ghi số trang hoặc mục bên cạnh mỗi thông tin, và ghi **TODO** khi chưa chắc.
7. Gửi TV1 link Overleaf và trang đó.

---

## 9. Cấu trúc báo cáo gợi ý

### Assignment 4: SOTA's Limitations
1. Mục tiêu và câu hỏi
2. Setup (dữ liệu, mô hình, cài đặt, GPU)
3. Thí nghiệm và phát hiện
   - 3.1 Seed làm thay đổi ảnh
   - 3.2 Độ nhất quán của chữ (điểm chính, CER, bất đồng giữa các seed)
   - 3.3 Trường hợp eduroam (hình)
   - 3.4 Ảnh hưởng của prompt
   - 3.5 Giới hạn bộ nhớ khi ảnh đầu vào lớn
4. Các phương pháp liên quan (một trang, từ bài báo)
5. Giới hạn của thí nghiệm và bước tiếp theo

### Assignment 5: Baseline
1. Cách làm (DoSSR, quy trình chạy)
2. Chi tiết cài đặt (bộ chạy, các bản vá, kiểm tra tính tương đương)
3. Setup thí nghiệm (dữ liệu, cài đặt, phần cứng, commit, cấu hình)
4. Kết quả: bảng 10 seed (PSNR, SSIM, LPIPS, trung bình ± độ lệch chuẩn), thời gian, bộ nhớ; điểm chữ nếu có
5. Phân tích và giới hạn
6. Khả năng chạy lại (lệnh, phiên bản, seed)

---

## 10. Quy tắc chung

- Lưu mọi thứ vào Drive chung (`lemmepass/`), đặt tên file rõ ràng.
- Xong một bước thì báo trong nhóm: "xong: việc gì, ở đâu".
- **Bị kẹt:** gửi đúng câu lệnh và **toàn bộ nội dung lỗi**. Đừng đoán, và đừng im lặng quá vài tiếng.
- Báo sớm nếu việc sẽ lâu hơn dự kiến. Báo ở ngày đầu là chuyện nhỏ, báo sát hạn là chuyện lớn.
- Chữ trong báo cáo phải phản ánh đúng dữ liệu. Nếu một giả thuyết không đúng, vẫn ghi lại đúng như vậy.
- Nếu dùng công cụ AI hỗ trợ viết code hoặc viết bài, ghi rõ và tuân theo quy định của môn học.

---

## 11. Phương án dự phòng

| Tình huống | Cách xử lý |
|---|---|
| Có quá ít ảnh có chữ đọc được | Ghi rõ số lượng chính xác và chỉ dùng tập đã đánh dấu. Chưa chuyển sang Real-CE x4 trong đợt này. |
| OCR không đọc được cả chữ trên ảnh nét | Ghi vào báo cáo. Đó là giới hạn của phép đo, và vẫn là một phát hiện. |
| Phát hiện còn thiếu hoặc sai ảnh trong 2.000 ảnh | Chạy lại đúng seed hoặc đúng ảnh bị thiếu (chạy lại sẽ bỏ qua ảnh đã xong). |
| Chấm điểm chữ chạy chậm trên CPU | Chạy trên Colab miễn phí, hoặc chạy trước trên các ảnh có chữ. |
| Hai file CSV gộp bị lỗi | Dùng báo lỗi của script gộp để quay lại đúng người sửa. |

---

## 12. Sau Assignment 4 (cho Assignment 5)

- **TV1:** bảng 10 seed hoàn chỉnh, bảng bật/tắt SDPA nếu còn thời gian, hỗ trợ chạy thử LoRA trên Colab.
- **TV2:** hoàn thiện điểm chữ, xem Real-CE x4 nếu cần.
- **TV3:** quay lại LoRA (kế hoạch ở `docs/lora_feasibility_and_loss_design.md`). Chú ý: dữ liệu huấn luyện LoRA không được trùng với ảnh đánh giá.
- **TV4:** bản nháp báo cáo baseline trước **thứ Sáu 13/11**.
- Nộp **tối thứ Hai 16/11**.

---

## 13. Danh sách kiểm tra trước khi nộp

- [ ] Đủ 2.000 ảnh, đúng kích thước 512x512
- [ ] File CSV nhãn đã gộp, không còn ô `readable` trống
- [ ] Điểm chữ và metrics đã chạy xong, có file CSV
- [ ] Hình eduroam đã đưa vào báo cáo
- [ ] Mọi số liệu trong báo cáo đều có trong log hoặc tài liệu
- [ ] Phần bài báo liên quan có ghi "theo bài báo" và số trang
- [ ] Có ghi commit, cấu hình, seed để chạy lại
- [ ] Tên file đúng `<StudentID1>_<StudentID2>_....pdf`
- [ ] File là PDF, mở được, các hình hiển thị đúng
- [ ] Nộp trước 22:00 thứ Hai, chụp màn hình xác nhận

---

## 14. Câu hỏi mở (TODO)

1. Mã số sinh viên của từng thành viên và quy tắc đặt tên file.
2. Số ảnh có chữ đọc được trong `test_HR` (chờ TV2 và TV3).
3. Công cụ OCR được chọn (chờ kết quả kiểm tra của TV3).
4. Hạn của lần chạy thử LoRA (hỏi mentor).
5. Dữ liệu huấn luyện LoRA lấy từ đâu (không được trùng ảnh đánh giá).
6. Prompt dùng cho 3 bản LoRA (đề xuất: prompt rỗng, giống lúc DoSSR huấn luyện; cần thống nhất với mentor).
7. Tên các thành viên và vai trò chính thức: TV1 = Hoàng Nguyên Anh (`25125002`), TV2 = Huỳnh Trần Gia Hân (`25125042`), TV3 = Hồ Quang Minh (`25125060`), TV4 = Nguyễn Phan Minh Nhật (`25125033`).

# Bước 8  Phân tích kết quả Memory Systems for AI Agent

## Môi trường, phép đo và cách chạy lại

Kết quả dưới đây được đo trên Windows bằng Python 3.14.7 của `.venv\Scripts\python.exe`. Chạy từ root repo:

```powershell
.\.venv\Scripts\python.exe src/benchmark.py
.\.venv\Scripts\python.exe scripts/analyze_memory.py
.\.venv\Scripts\python.exe -m pytest src -q
```

Benchmark mặc định dùng `compact_threshold_tokens=512`, `compact_keep_messages=6`. Script đối chứng dùng cùng hai giá trị đó cho Advanced mặc định; riêng Advanced tắt compact đặt threshold `1_000_000_000` và vẫn cấu hình `compact_keep_messages=6`. Vì threshold này không bị vượt trong hai dataset, không có compaction và toàn bộ messages của mỗi thread được giữ, không bị cắt xuống 6. Giá trị 6 chỉ quyết định số message gần nhất giữ nguyên văn **khi compaction xảy ra**. Ở cả hai dataset, trường hợp tắt compact thực đo được **0 compactions**. Script lấy các hằng số mặc định từ `config.py`, tạo `LabConfig` với provider stub và ép offline; benchmark mặc định in cấu hình thực tế sau khi nạp config. Khi chạy trong môi trường có override ngưỡng, cần đối chiếu dòng `Compact settings` với 512/6 trước khi so hai lệnh.

Hai lệnh đo tạo namespace mới dưới `state/`; test dùng thư mục tạm riêng. Trong mỗi dataset, Baseline, Advanced mặc định và Advanced tắt compact có agent instance và state riêng; Standard và Stress cũng độc lập. Profile được giữ trên đĩa tại `state/<namespace>/<standard|stress>/<agent>/profiles/` để kiểm tra. Không xóa hoặc dùng lại profile của lần chạy trước. Các agent chạy offline tất định; không tạo LLM, không gọi API hoặc judge thật.

Hai dataset giữ nguyên turns, thứ tự và câu hỏi. Mỗi conversation có training thread và recall thread khác nhau. Một Advanced instance được dùng suốt một dataset để profile của cùng user đi qua các conversation; `expected_contains` chỉ dùng khi chấm, không đưa cho agent. Tổng `Agent tokens only` cộng số token **mỗi lượt** của response assistant; `Prompt tokens processed` cộng số token prompt **mỗi lượt** của tất cả lần `reply()`, gồm training và recall. Estimator đếm gần đúng theo độ dài ký tự (`len(text.strip()) // 4`, tối thiểu 1 cho chuỗi không rỗng), không phải token usage do provider xác nhận. Prompt Baseline là lịch sử role-labelled của thread tính đến user message hiện tại; prompt Advanced là nội dung `User.md` + summary + recent messages có role, đo trước khi thêm response hiện tại. Nhánh offline không có system prompt.

`Memory growth (bytes)` là tổng kích thước `User.md` **sau trừ trước** cho từng user duy nhất trong một lần chạy; `Compactions` cộng count của từng thread duy nhất. Recall của một câu là 0 nếu không khớp, 0,5 nếu khớp một phần, 1 nếu đủ mọi `expected_contains`. Quality là tỷ lệ literal expected facts khớp trong response, cũng trên thang 0–1. Hai điểm được lấy trung bình trên những câu có expected không rỗng; câu expected rỗng vẫn được hỏi và tính token nhưng không vào mẫu số điểm.

## Kết quả mặc định đã đo

Chạy `src/benchmark.py` trên namespace mới `state/benchmark-ddwuz4bg`. Các cột token có đơn vị **token ước lượng**, growth là **byte thực**; recall và quality là phần trăm.

### Standard Benchmark — `data/conversations.json`

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1279 | 15827 | 0.0% | 0.0% | 0 | 0 |
| Advanced | 1182 | 22167 | 100.0% | 100.0% | 339 | 0 |

### Long-Context Stress Benchmark — `data/advanced_long_context.json`

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 272 | 22655 | 0.0% | 0.0% | 0 | 0 |
| Advanced | 276 | 10377 | 100.0% | 100.0% | 283 | 26 |

## Đối chứng: Advanced tắt compact

`scripts/analyze_memory.py` gọi lại `load_conversations()`, `run_agent_benchmark()` và `format_rows()` cho ba trường hợp. Nó không chép logic scoring hoặc sửa input. Hai lần chạy trên `state/analyze-memory-e3h2s_an` và `state/analyze-memory-miy4l_tm` tạo các profile riêng. **Mọi giá trị số thô trước làm tròn đều trùng nhau**: output, prompt, recall/quality dạng float (0.0 hoặc 1.0), byte profile và compactions. Chỉ tên namespace khác.

### Standard — cùng input và phạm vi đo

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1279 | 15827 | 0.0% | 0.0% | 0 | 0 |
| Advanced mặc định | 1182 | 22167 | 100.0% | 100.0% | 339 | 0 |
| Advanced tắt compact | 1182 | 22167 | 100.0% | 100.0% | 339 | 0 |

### Stress — cùng input và phạm vi đo

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 272 | 22655 | 0.0% | 0.0% | 0 | 0 |
| Advanced mặc định | 276 | 10377 | 100.0% | 100.0% | 283 | 26 |
| Advanced tắt compact | 276 | 23474 | 100.0% | 100.0% | 283 | 0 |

Với cùng một chỉ số, **overhead** của Advanced so với Baseline là `(Advanced − Baseline) / Baseline × 100%`. **Reduction** của bản compact so với một mốc tham chiếu là `(reference − compact) / reference × 100%`. Tính trên số nguyên chưa làm tròn:

| So sánh Prompt tokens processed | Phép tính | Kết quả |
| --- | ---: | ---: |
| Standard: Advanced mặc định so với Baseline | `(22167 − 15827) / 15827` | **+40,06% overhead** |
| Stress: Advanced tắt compact so với Baseline | `(23474 − 22655) / 22655` | **+3,62% overhead** |
| Stress: Advanced mặc định so với Baseline | `(22655 − 10377) / 22655` | **54,20% reduction** |
| Stress: Advanced mặc định so với Advanced tắt compact | `(23474 − 10377) / 23474` | **55,79% reduction** |

Chỉ phép so **Advanced mặc định với Advanced tắt compact** giữ nguyên agent, profile và scoring để cô lập tác động của cấu hình compact. Con số 54,20% so Baseline cho thấy chênh lệch giữa hai hệ thống hoàn chỉnh, gồm cả khác biệt memory và cách sinh response.

## Bốn câu hỏi của Bước 8

### 1. Vì sao Advanced recall tốt hơn Baseline?

Cross-session recall là **100% so với 0%** ở cả Standard và Stress. Advanced trích fact từ phát biểu user bằng `extract_profile_updates()`, lọc assertion trong `_asserted_updates()`, rồi `upsert_fact()` theo khóa vào `state/.../profiles/<user_id>/User.md`. Khi hỏi ở recall thread mới, `_offline_response()` đọc lại các fact trong profile. Profile tăng **339 byte** ở Standard và **283 byte** ở Stress. Baseline chỉ giữ danh sách message theo `thread_id` trong instance, nên recall thread mới không mang lịch sử training; nó không tạo profile, growth bằng 0.

Profile bền vững theo user và summary/recent theo thread là **hai lớp khác nhau**. Điều này giải thích tại sao Standard Advanced đạt recall 100% dù có **0 compactions**. Kết quả đo trên những câu recall và fact hiện có trong hai dataset; nó không chứng minh agent sẽ nhận ra mọi cách diễn đạt ngoài dữ liệu.

### 2. Vì sao Advanced có thể tốn hơn trong hội thoại ngắn?

Ở Standard, prompt Advanced là **22.167** so với **15.827** token ước lượng của Baseline, overhead **40,06%**. Cả hai cấu hình Advanced đều có 0 compactions và đúng cùng số prompt, nên compact chưa bù chi phí context. `_estimate_prompt_context_tokens()` cộng cả `User.md`, summary và recent messages; profile xuất hiện trong prompt qua các lượt sau khi được ghi. Việc ghi file là thao tác lưu trữ, **không tự phát sinh token LLM**.

`Agent tokens only` là phần assistant sinh, Standard Advanced **1.182**, thấp hơn Baseline **1.279** (giảm **7,58%**); không thể dùng cột này để nói Advanced tốn output hơn. Overhead ở đây thuộc **Prompt tokens processed**. Mức overhead phụ thuộc độ dài profile, số lượt và cấu trúc câu hỏi của dataset này; không phải hằng số cho mọi hội thoại ngắn.

### 3. Vì sao compact có lợi trong hội thoại dài?

Stress Advanced mặc định có **26 compactions**, prompt **10.377**; Advanced tắt compact có **0 compactions**, prompt **23.474**. Giảm **55,79%** khi so trong cùng thiết kế Advanced. So với Baseline **22.655**, bản compact giảm **54,20%**, nhưng phép so này còn bao gồm khác biệt giữa hai agent. Output Stress Advanced là **276** ở cả hai cấu hình, và recall/quality cùng **100%** trong dữ liệu đo; lợi ích quan sát được tập trung ở prompt context.

`CompactMemoryManager.append()` kiểm tra token của summary và message thread. Khi vượt threshold và có nhiều hơn 6 message, nó thay phần cũ bằng summary và giữ 6 message gần nhất nguyên văn. Advanced đo prompt gồm `User.md` + summary + recent tại thời điểm trước response, nên không mang toàn bộ lịch sử như Baseline. Summary chỉ giữ keyed facts và tối đa 6 topic gần đây, với độ dài mỗi fact/topic bị giới hạn; vì vậy có rủi ro mất chi tiết. Threshold **512** là điều kiện kích hoạt, không phải cam kết prompt luôn dưới 512: riêng 6 recent messages hoặc profile có thể đã vượt ngưỡng.

### 4. Memory tăng trưởng ra sao, rủi ro gì?

Profile tăng **339 byte Standard**, **283 byte Stress** trong từng lần benchmark khởi đầu sạch; hai số này là tổng byte `User.md` sau trừ trước, không phải tốc độ tăng theo thời gian thực. `upsert_fact()` thay giá trị cũ của cùng khóa, vì vậy correction không tạo thêm khóa trùng, nhưng fact mới vẫn làm profile lớn hơn. Standard có 0 compactions mà profile vẫn tăng; Stress có 26 compactions nhưng profile vẫn tăng 283 byte. Compact xử lý lịch sử **thread**, không nén `User.md`.

Rủi ro cụ thể là regex extraction có thể gán sai fact nếu câu ngoài các mẫu đã kiểm thử, hoặc bỏ sót correction diễn đạt khác. Profile không có thời hạn hết hiệu lực, confidence score hay cơ chế tự xóa, nên fact từng đúng có thể lỗi thời. Summary giới hạn số topic và cắt ngắn text, nên ngữ cảnh thread cũ có thể mất chi tiết sau nhiều lần nén. `User.md` là UTF-8 trên đĩa, chứa thông tin cá nhân dưới dạng đọc được; repo bỏ qua `state/` khi commit, nhưng implementation chưa có chính sách lưu giữ hoặc mã hóa profile. Hai dataset nhỏ không định lượng được các rủi ro này theo thời gian dài.

## Cải thiện đã có và bằng chứng

Implementation hiện dùng **structured facts theo khóa** (`name`, `location`, `profession`, `response_style`, v.v.) và `upsert_fact()`; correction thay giá trị cũ. Bộ lọc assertion bỏ câu hỏi recall, câu giả định/đùa và một số ngữ cảnh như địa điểm đi họp; Advanced cũng chỉ lấy message **user** làm nguồn fact. Test `test_correction_replaces_old_fact_without_meeting_or_joke_noise`, `test_users_are_isolated_and_questions_do_not_create_facts` và các test extraction kiểm chứng những trường hợp này. Chúng giảm nguy cơ ghi fact không được user khẳng định, nhưng là heuristic regex nên vẫn có thể sai với câu mới.

Correction style được xử lý theo thuộc tính: yêu cầu một đoạn văn/không dùng bullet loại bỏ bullet cũ; “không cần ưu tiên trade-off nữa” bỏ ưu tiên đó, trong khi thuộc tính không xung đột còn lại được giữ. `test_explicit_style_corrections_remove_conflicting_traits_and_persist` kiểm tra cả profile và response ở thread/instance mới. Cơ chế này tránh profile chứa các chỉ dẫn style mâu thuẫn, nhưng chỉ nhận ra những mẫu correction đã code; không có mô hình hiểu ý định tổng quát.

Benchmark chung cho thấy Advanced recall 100%; các test profile kiểm tra correction thay giá trị cũ theo khóa. **Chưa có đối chứng tắt riêng từng bộ lọc, upsert hoặc xử lý style**, nên không gán cho từng cải thiện một mức tăng recall hay mức giảm token định lượng. Cấu hình tắt compact ở trên chỉ đo riêng ảnh hưởng của compact. Không triển khai confidence threshold hoặc memory decay trong mốc này.

## Giới hạn đánh giá

`expected_contains` và matching literal sau chuẩn hóa Unicode/case/whitespace quyết định recall; quality là tỷ lệ expected facts xuất hiện trong câu trả lời, **liên quan trực tiếp đến recall**, không phải judge độc lập về độ tự nhiên, đúng ngữ cảnh hoặc an toàn. Điểm 100% trên hai dataset nhỏ không chứng minh hiểu ngôn ngữ tổng quát. Offline response chọn câu trả lời bằng heuristic theo từ khóa và profile/summary/recent; token estimator dựa độ dài ký tự. Live workflow chưa triển khai, nên chưa kiểm chứng provider token usage, chất lượng LLM hay chi phí thực; không thể chuyển trực tiếp token ước lượng thành USD.

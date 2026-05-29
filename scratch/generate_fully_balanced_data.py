import os
import csv
import random
import pandas as pd

additional_csv_path = r"d:\XuanDuc\TaiLieuKi8\CuoiKiHocSau\data\additional_train.csv"
train_csv_path = r"d:\XuanDuc\TaiLieuKi8\CuoiKiHocSau\data\train.csv"

# Load original additional_train.csv and extract only label 0 (real)
print("Loading additional_train.csv...")
df_additional = pd.read_csv(additional_csv_path)
# Keep only label 0 (Real) from the original file (which is 3320 rows)
df_real_additional = df_additional[df_additional['label'] == 0]
real_count = len(df_real_additional)
print(f"Loaded {real_count} real news (label 0) from additional_train.csv.")

# Load train.csv to get the counts
df_train = pd.read_csv(train_csv_path)
train_real_count = len(df_train[df_train['label'] == 0])
train_fake_count = len(df_train[df_train['label'] == 1])
print(f"train.csv distribution: Real (0) = {train_real_count}, Fake (1) = {train_fake_count}")

# Calculate target fake count for additional_train.csv
total_real_target = train_real_count + real_count  # 7269 + 3320 = 10589
target_fake_additional = total_real_target - train_fake_count  # 10589 - 1472 = 9117
print(f"Target fake news count to generate: {target_fake_additional}")

# Define rich templates for all themes to match the writing style of train.csv

# Theme 1: Covid-19 Prevention & Cure Myths
c1_intros = [
    "Lời khuyên từ bác sĩ {doctor} đầu ngành y tế Việt Nam:",
    "Thông tin chia sẻ từ chuyên gia dịch tễ học Lâm sàng {doctor}:",
    "Bác sĩ {doctor} từ bệnh viện lớn tại Canada vừa gửi thông tin hữu ích:",
    "Mọi người truyền tai nhau bài thuốc nam trị virus cực hay của lương y {doctor}:",
    "Bài thuốc dân gian gia truyền của danh y {doctor} giúp tiêu diệt dứt điểm dịch bệnh:",
    "Phương pháp tự điều trị virus tại nhà cực kỳ đơn giản và hiệu quả do bác sĩ {doctor} chia sẻ:",
    "Chia sẻ của bác sĩ Việt kiều {doctor} đang trực tiếp điều trị tại ổ dịch dã chiến:",
    "Mẹo bảo vệ sức khỏe gia đình trước biến thể virus mới nguy hiểm theo BS {doctor}:"
]

c1_methods = [
    "xông mũi họng bằng hỗn hợp sả, gừng, tỏi tươi đun sôi liên tục {num} phút",
    "tẩm nước muối sinh lý ấm nồng độ {num}% vào mặt ngoài của khẩu trang y tế",
    "uống một cốc nước tỏi tươi giã nhuyễn đun nóng vào mỗi buổi sáng khi đói liên tục {num} ngày",
    "nhỏ {num} giọt nước cốt chanh nguyên chất trực tiếp vào mắt và niêm mạc mũi",
    "súc miệng súc họng bằng cồn sát khuẩn {num} độ pha với nước ấm",
    "uống nước ấm ở nhiệt độ khoảng {num} độ C đều đặn mỗi 15 phút một lần",
    "thoa dầu gió Trường Sơn vào hai bên thái dương, lỗ tai và lòng bàn chân trong {num} ngày liền",
    "xông phòng ngủ bằng quả bồ kết khô hoặc tinh dầu sả chanh nguyên chất trong {num} giờ",
    "uống nước gừng tươi pha mật ong và {num} lát hành tây thái mỏng mỗi tối",
    "thường xuyên nhai {num} tép tỏi sống và ngậm nước muối ấm pha gừng"
]

c1_effects = [
    "sẽ giúp diệt sạch hoàn toàn 100% virus Corona Vũ Hán trong vòng {seconds} giây.",
    "ngăn ngừa tuyệt đối không cho virus bám vào niêm mạc khí quản và phổi sau {days} ngày thực hiện.",
    "kích thích cơ thể tự sản sinh lượng lớn kháng thể T-cell để đề kháng mọi biến thể trong vòng {days} tuần.",
    "lập tức làm bất hoạt và phân hủy lớp vỏ protein của virus sau {seconds} giây tiếp xúc.",
    "chữa khỏi dứt điểm hoàn toàn các triệu chứng ho sốt chỉ sau {days} ngày duy nhất.",
    "tạo ra màng lọc bảo vệ phổi tuyệt đối khỏi sự xơ hóa do virus gây ra chỉ trong {days} ngày.",
    "giúp thanh lọc hoàn toàn đường hô hấp và đào thải độc tố virus ra ngoài sau {seconds} giây.",
    "đã được các nhà khoa học Nhật Bản kiểm nghiệm và chứng minh hiệu quả sau {days} ngày dùng thử."
]

c1_ctas = [
    "Hãy chia sẻ gấp thông tin hữu ích này để chung tay cứu sống gia đình và bạn bè!",
    "Bấm nút chia sẻ ngay lập tức vì sức khỏe của cả cộng đồng Việt Nam!",
    "Mọi người hãy lưu lại bài viết này và áp dụng ngay cho con em mình nhé!",
    "Cứu một mạng người hơn xây tòa tháp, chia sẻ nhanh tay mọi người ơi!",
    "Hãy lan tỏa thông điệp này đến thật nhiều nhóm Facebook để giúp đỡ mọi người!",
    "Thông tin quý hơn vàng, đừng giữ cho riêng mình, hãy chia sẻ ngay hôm nay!"
]

# Theme 2: Covid-19 Cover-ups & Local Rumors
c2_locations = [
    "tại tỉnh Bạc Liêu", "tại địa bàn Bắc Ninh", "tại thành phố Hải Phòng", "ở trung tâm Đà Nẵng",
    "tại thành phố du lịch Nha Trang", "tại khu công nghiệp VSIP Bình Dương", "ở nhà máy Samsung Thái Nguyên",
    "tại bệnh viện Bạch Mai Hà Nội", "tại quận 1 thành phố Hồ Chí Minh", "ở khu chung cư Linh Đàm Hà Nội",
    "tại huyện Bình Chánh TPHCM", "tại thành phố Biên Hòa Đồng Nai", "tại ổ dịch Mê Linh Hà Nội",
    "tại quận Cầu Giấy Hà Nội", "tại quận Gò Vấp TP.HCM", "ở thành phố Thuận An Bình Dương"
]

c2_rumors = [
    "đã ghi nhận {cases} ca tử vong đầu tiên do virus nhưng cơ quan chức năng đang ém thông tin để tránh hoang mang dư luận.",
    "phát hiện {cases} công nhân xét nghiệm dương tính nhưng ban giám đốc giấu kín để không bị đóng cửa nhà máy.",
    "hàng chục người nghi nhiễm virus đã trốn khỏi khu cách ly tập trung {loc} và đang di chuyển tự do.",
    "phát hiện ổ dịch siêu lây nhiễm với {cases} ca mắc mới tại chợ đầu mối nhưng cơ quan chức năng chưa công bố rộng rãi.",
    "phát hiện trường hợp người dân bất ngờ ngất xỉu và tử vong ngay trên vỉa hè do suy hô hấp cấp vì nhiễm virus tại {loc}.",
    "bệnh viện địa phương đã rơi vào tình trạng quá tải nghiêm trọng, số ca tử vong tăng lên {cases} người vượt quá tầm kiểm soát.",
    "chính quyền chuẩn bị ban bố lệnh phong tỏa toàn diện trong vòng {days} ngày tới, người dân cần khẩn trương tích trữ thực phẩm."
]

c2_urgencies = [
    "Tình hình đang cực kỳ nguy kịch và nghiêm trọng, mọi người hãy tự bảo vệ lấy gia đình mình!",
    "Đừng tin vào những bản tin xoa dịu của báo chí, hãy tự nâng cao cảnh giác!",
    "Thông tin đang bị kiểm duyệt gắt gao, mọi người chia sẻ nhanh kẻo bài viết bị xóa!",
    "Cảnh báo đỏ cho tất cả mọi người, hạn chế ra đường tối đa trong thời gian này!",
    "Đọc tin mà rùng mình ghê sợ, mọi người lưu ý và đề phòng cẩn thận nhé!",
    "Hãy chia sẻ cho người thân ở khu vực lân cận biết để tránh xa ổ dịch!"
]

# Theme 3: Conspiracy Theories
c3_intros = [
    "Nguồn tin rò rỉ chấn động từ cựu nhân viên tình báo cấp cao {name}:",
    "Báo chí độc lập {source} vừa đăng tải phóng sự điều tra vạch trần:",
    "Nhiều tài liệu mật của chính phủ các nước vừa bị các hacker {name} phát tán trên mạng xã hội:",
    "Các nhà khoa học độc lập hàng đầu tại {source} đã đưa ra bằng chứng:",
    "Sự thật kinh hoàng đằng sau đại dịch toàn cầu năm {year} mà các tập đoàn truyền thông lớn đang che giấu:"
]

c3_claims = [
    "virus SARS-CoV-2 thực chất là một loại vũ khí sinh học nguy hiểm được sản xuất trong phòng thí nghiệm để phát động chiến tranh tiền tệ.",
    "vaccine ngừa Covid-19 chứa các vi chíp điện tử siêu nhỏ nhằm mục đích giám sát và điều khiển hành vi con người qua mạng 5G của tập đoàn {corp}.",
    "toàn bộ đại dịch này là một âm mưu được dàn dựng bởi thế lực ngầm {corp} nhằm giảm dân số thế giới và định hình lại trật tự mới.",
    "Tổ chức Y tế Thế giới WHO đã cấu kết với các tập đoàn dược phẩm khổng lồ {corp} để thổi phồng mức độ nguy hiểm nhằm trục lợi.",
    "virus được thiết kế nhân tạo để nhắm mục tiêu phá hủy hệ thống tài chính toàn cầu dưới sự chỉ đạo của tổ chức {corp}."
]

# Theme 4: Protests, Government, Utilities & Court Rumors
c4_subjects = [
    "Lực lượng Cảnh sát Cơ động dắt theo chó nghiệp vụ và công cụ hỗ trợ",
    "Hội đồng thẩm phán Tòa án nhân dân",
    "Cơ quan Cảnh sát điều tra Bộ Công an",
    "Ban lãnh đạo Tổng công ty Điện lực EVN",
    "Thành viên hội đồng quản trị ngân hàng SCB và tập đoàn Vạn Thịnh Phát",
    "Các cơ quan tư pháp và viện kiểm sát địa phương",
    "Đội quản lý trật tự đô thị phối hợp cùng lực lượng chức năng"
]

c4_accusations = [
    "đã sử dụng các biện pháp bức cung, nhục hình dã man để ép buộc bị can {citizen} nhận tội thay cho con em của {official}.",
    "cố ý làm sai lệch hoàn toàn biên bản hiện trường vụ tai nạn giao thông nghiêm trọng của {citizen} nhằm bao che cho người thân của {official}.",
    "âm thầm phê duyệt đề án tăng giá dịch vụ lên gấp nhiều lần nhằm bắt người dân gánh nợ cho các khoản thua lỗ {money} tỷ đồng.",
    "tiến hành cưỡng chế và bắt giữ trái phép công dân {citizen} phản đối dự án thu hồi đất bẩn để giao đất vàng cho doanh nghiệp của {official}.",
    "sử dụng quyền lực chính trị để can thiệp trắng trợn vào kết quả xét xử của tòa án nhằm chạy án {money} tỷ cho các tội phạm tham nhũng.",
    "tiêu hủy các chứng cứ quan trọng liên quan đến sai phạm quản lý đất đai của {official} hòng trốn tránh trách nhiệm trước thanh tra.",
    "thu giữ tài sản trị giá {money} triệu đồng và đập phá sạp hàng của tiểu thương {citizen} mà không có quyết định hành chính hợp pháp."
]

c4_sentiments = [
    "Thật sự đau xót cho một nền tư pháp nước nhà, công lý giờ chỉ dành cho kẻ có tiền!",
    "Người dân nghèo thấp cổ bé họng biết kêu oan với ai trước sự áp bức bẩn tưởi này?",
    "Hành vi vi hiến, lộng hành của nhóm lợi ích sớm muộn gì cũng sẽ phải trả giá đắt trước công luận!",
    "Mọi người hãy đồng lòng chia sẻ và lên tiếng phản đối mạnh mẽ hành vi sai trái này!",
    "Đâu rồi lương tâm của những người mang danh bảo vệ công lý và pháp luật?",
    "Hãy chia sẻ rộng rãi để đưa sự thật này ra ánh sáng, cứu lấy những người vô tội!"
]

# Theme 5: Scams, Giveaways & Authority Impersonation
c5_warning_words = [
    "CẢNH BÁO KHẨN CẤP:", "THÔNG BÁO QUAN TRỌNG:", "LỆNH TRIỆU TẬP KHẨN CẤP:", "QUYẾT ĐỊNH XỬ PHẠT HÀNH CHÍNH:",
    "LỆNH BẮT GIAM TẠM THỜI:", "CẢNH BÁO AN NINH MẠNG:", "YÊU CẦU PHỐI HỢP ĐIỀU TRA KHẨN CẤP:",
    "THÔNG BÁO KHÓA THUÊ BAO VĨNH VIỄN:", "LỆNH TRUY NÃ ĐẶC BIỆT THI HÀNH CƯỠNG CHẾ:", "THÔNG BÁO PHẠT NGUỘI GIAO THÔNG:"
]

c5_agencies = [
    "Bộ Công an", "Viện Kiểm sát Nhân dân Tối cao", "Cơ quan Cảnh sát Điều tra Công an Thành phố Hà Nội",
    "Cơ quan Cảnh sát Điều tra Công an TP.HCM", "Cục Cảnh sát Giao thông", "Tổng cục Thuế Việt Nam",
    "Bảo hiểm Xã hội Việt Nam", "Tổng công ty Điện lực Việt Nam EVN", "Bộ Y tế", "Bộ Thông tin và Truyền thông",
    "Cục An ninh mạng và phòng chống tội phạm công nghệ cao", "Sở Giáo dục và Đào tạo", "Tổng công ty Cấp nước SAWACO"
]

c5_accusations = [
    "tài khoản của ông/bà đang nằm trong danh sách đen nghi vấn rửa tiền và tài trợ khủng bố quốc tế",
    "số Căn cước công dân của bạn liên quan trực tiếp đến đường dây buôn lậu ma túy xuyên quốc gia",
    "phát hiện hành vi trốn thuế thu nhập cá nhân và sử dụng trái phép hóa đơn tài chính giả",
    "phương tiện giao thông của bạn đã vượt đèn đỏ và gây tai nạn nghiêm trọng rồi bỏ chạy tại giao lộ",
    "đang có hành vi vi phạm nghiêm trọng Luật An ninh mạng khi chia sẻ thông tin chống phá Nhà nước",
    "số điện thoại thuê bao của bạn phát tán hàng ngàn tin nhắn rác lừa đảo chiếm đoạt tài sản",
    "liên quan đến hồ sơ đen của vụ án chiếm đoạt tài sản quy mô lớn tại tập đoàn Vạn Thịnh Phát và SCB",
    "chưa hoàn thành nghĩa vụ đóng tiền điện nước kỳ gần nhất và đang có nợ đọng quá hạn kéo dài",
    "hệ thống phát hiện thông tin sinh trắc học và thẻ Căn cước công dân gắn chíp của bạn bị lỗi đồng bộ",
    "tài khoản ngân hàng của bạn đang giao dịch mờ ám với các trang web cá độ bóng đá và cờ bạc trực tuyến"
]

c5_bank_names = ["Vietcombank", "Techcombank", "BIDV", "Agribank", "Vietinbank", "MB Bank", "Sacombank", "VPBank", "TPBank", "ACB"]

c5_call_to_actions = [
    "yêu cầu quý khách truy cập ngay cổng dịch vụ công trực tuyến tại {fake_url} để cập nhật thông tin phục vụ xác minh",
    "đề nghị tải và cài đặt ngay ứng dụng giám sát bảo mật 'Bộ Công An' tại đường link {fake_url} để quét khuôn mặt đối soát",
    "yêu cầu khẩn trương chuyển toàn bộ số dư tiền tiết kiệm hiện có sang tài khoản giám sát an toàn số {bank_acc} tại ngân hàng {bank_name} để kiểm toán tài chính",
    "yêu cầu nhấp vào liên kết {fake_url} để điền thông tin đăng nhập ngân hàng và mã xác thực OTP nhằm đối soát dòng tiền nghi vấn",
    "yêu cầu liên hệ khẩn cấp với Đại úy điều tra viên qua số hotline {phone} để làm việc gián tiếp và giữ bí mật thông tin",
    "vui lòng đóng phạt nguội số tiền {money} qua cổng dịch vụ trực tuyến tại link {fake_url} để hoàn thành nghĩa vụ thanh toán",
    "hãy truy cập trang liên kết chính phủ {fake_url} để điền tờ khai bảo mật và cung cấp mã OTP ngân hàng xác thực tài khoản"
]

c5_penalties = [
    "Nếu trì hoãn hoặc bất hợp tác, cơ quan công an sẽ tiến hành phong tỏa tài khoản vĩnh viễn và khởi tố hình sự.",
    "Quá thời hạn 2 giờ kể từ khi nhận thông báo này, lệnh cưỡng chế và bắt tạm giam 3 tháng sẽ được thi hành.",
    "Mọi hành vi trốn tránh sẽ bị xử lý nghiêm khắc trước pháp luật và gửi lệnh triệu tập về nơi cư trú hoặc làm việc.",
    "Quá hạn nộp phạt, chúng tôi sẽ tiến hành tước quyền sử dụng giấy phép lái xe vĩnh viễn và niêm phong phương tiện vi phạm.",
    "Hết thời hạn quy định, chúng tôi sẽ tự động tạm ngắt dịch vụ cấp điện nước và thu hồi hợp đồng sử dụng đối với hộ gia đình."
]

c5_promo_brands = [
    "Vietnam Airlines", "Shopee Việt Nam", "Ví MoMo", "Tập đoàn Vingroup", "Tổng công ty Viettel",
    "Thế Giới Di Động", "Facebook Việt Nam", "Ngân hàng Vietcombank", "Ngân hàng Techcombank",
    "Samsung Việt Nam", "Heineken Việt Nam", "Coca-Cola Việt Nam", "Hãng Vietjet Air", "Grab Việt Nam"
]

c5_reasons = [
    "nhân dịp kỷ niệm ngày thành lập tập đoàn", "trong chương trình tri ân khách hàng thân thiết năm nay",
    "nhân ngày hội mua sắm khuyến mãi lớn nhất trong năm", "hệ thống đã lựa chọn ngẫu nhiên tài khoản của bạn",
    "nhân dịp đạt cột mốc 20 triệu người dùng hoạt động trên toàn quốc"
]

c5_prizes = [
    "1 chiếc xe máy Honda SH 150i cùng 50 triệu đồng tiền mặt",
    "1 điện thoại thông minh iPhone 15 Pro Max và voucher mua sắm 5 triệu",
    "phiếu mua hàng siêu thị trị giá 10 triệu đồng áp dụng mua sắm miễn phí",
    "gói quà tặng thanh toán hóa đơn đa năng trị giá 2.000.000đ cộng trực tiếp vào ví",
    "2 vé máy bay khứ hồi miễn phí đi bất kỳ chặng bay nội địa nào"
]

c5_scam_actions = [
    "quý khách vui lòng nhấp vào đường link {fake_url} để điền thông tin cá nhân và đăng ký nhận quà giao tận nhà.",
    "truy cập ngay trang sự kiện chính thức {fake_url} để đăng nhập tài khoản ngân hàng trực tuyến nhận tiền mặt tức thì.",
    "yêu cầu thanh toán lệ phí làm hồ sơ nhận giải là {money} bằng cách nạp thẻ cào điện thoại qua liên kết {fake_url}.",
    "vui lòng click vào liên kết {fake_url} để xác thực số điện thoại và chia sẻ tin nhắn này đến 5 nhóm Facebook để hoàn tất thủ tục."
]

# Helper lists for placeholders
doctors = ["Nguyễn Văn Thọ", "Trần Thị Kim Anh", "Phạm Hùng Cường", "Lê Thị Bích Vân", "Vũ Đức Minh", "Hoàng Văn Tuấn", "Bùi Huy Toàn", "Đặng Quang Huy"]
sources = ["báo Canada News", "trung tâm nghiên cứu Đức", "tạp chí Y khoa Pháp", "viện khoa học Đài Loan", "trang tin tức Hoa Kỳ"]
names = ["Nguyễn Văn Nam", "Trần Minh Hoàng", "Lê Hồng Đăng", "Phạm Quang Hải", "Vũ Huy Hoàng", "Đặng Tiểu Bình"]
corps = ["Bill Gates Foundation", "Rockefeller", "Pfizer", "Moderna", "Rothschild Group", "Big Pharma"]
citizens = ["Nguyễn Văn Hùng", "Trần Thị Mai", "Lê Văn Sơn", "Phạm Duy Khánh", "Đỗ Quang Vinh", "Vũ Đức Giang"]
officials = ["Chủ tịch UBND tỉnh", "Bí thư Tỉnh ủy", "Giám đốc Sở", "Thứ trưởng Bộ", "Cục trưởng Cục"]

def get_fake_url(agency_mode=True):
    sub = random.choice(["bocongan", "csgt", "gdt", "bhxh", "evn", "chuyentiengov", "bo-cong-an", "phatnguoi-csgt", "xacminh-taikhoan", "dichvucong-gov"]) if agency_mode else \
          random.choice(["shopee-trian", "vietnamairlines-gift", "momo-nhanloc", "viettel-trungthuong", "vinfast-feliz", "thegioididong-quatang", "facebook-nhangiai", "vcb-digibank", "samsung-s24", "heineken-he", "grab-khuyenmai"])
    domain = random.choice([".com", ".net", ".org", ".cc", ".xyz", ".info", ".net.vn", ".co", ".online", ".vip"])
    path = random.choice(["/login", "/auth", "/trian", "/nhangiai", "/xacminh", "/home", "/service", "/register"])
    return f"http://{sub}{domain}{path}"

def get_fake_phone():
    prefix = random.choice(["09", "08", "07", "03", "05"])
    return prefix + "".join(str(random.randint(0, 9)) for _ in range(8))

def get_fake_acc():
    return "".join(str(random.randint(0, 9)) for _ in range(random.choice([10, 12, 13])))

def get_fake_money():
    return f"{random.choice([50, 100, 200, 500, 800, 1000])}.000đ"

# Generate dataset safely in a single loop
generated_fake_set = set()
print(f"Generating exactly {target_fake_additional} unique, diverse Vietnamese fake news sentences...")

# Track distribution to ensure variety
counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
target_per_theme = target_fake_additional // 5

while len(generated_fake_set) < target_fake_additional:
    # Pick a theme that hasn't exceeded its share, unless all are filled up to quota
    available_themes = [t for t, count in counts.items() if count < target_per_theme]
    if not available_themes:
        available_themes = [1, 2, 3, 4, 5]
    
    theme = random.choice(available_themes)
    sentence = ""
    
    if theme == 1:
        intro = random.choice(c1_intros).format(doctor=random.choice(doctors))
        method = random.choice(c1_methods).format(num=random.choice([5, 10, 15, 20, 30, 2, 3, 7]))
        effect = random.choice(c1_effects).format(seconds=random.choice([30, 45, 60, 90, 120]), days=random.choice([2, 3, 5, 7]), days_pl=random.choice([1, 2, 3]))
        cta = random.choice(c1_ctas)
        sentence = f"{intro} Việc {method} {effect} {cta}"
        
    elif theme == 2:
        loc = random.choice(c2_locations)
        rumor = random.choice(c2_rumors).format(cases=random.randint(5, 500), loc=loc, days=random.choice([7, 10, 14, 21]))
        urg = random.choice(c2_urgencies)
        sentence = f"Tin nóng: Tình hình dịch bệnh {loc} {rumor} {urg}"
        
    elif theme == 3:
        intro = random.choice(c3_intros).format(name=random.choice(names), source=random.choice(sources), year=random.choice([2020, 2021, 2022, 2026]))
        claim = random.choice(c3_claims).format(corp=random.choice(corps))
        prefix = random.choice(["Đừng coi thường!", "Hãy tỉnh táo lên!", "Sự thật cuối cùng đã được phơi bày!"])
        sentence = f"{intro} {claim} {prefix}"
        
    elif theme == 4:
        sub = random.choice(c4_subjects)
        acc = random.choice(c4_accusations).format(
            citizen=random.choice(citizens),
            official=random.choice(officials),
            money=random.choice(["10", "20", "50", "100", "500", "hàng chục", "hàng trăm"])
        )
        sent = random.choice(c4_sentiments)
        sentence = f"Thông tin xã hội: {sub} {acc} {sent}"
        
    elif theme == 5:
        if random.random() < 0.5:
            # Authority Impersonation
            warn = random.choice(c5_warning_words)
            agency = random.choice(c5_agencies)
            acc = random.choice(c5_accusations)
            cta = random.choice(c5_call_to_actions).format(
                fake_url=get_fake_url(agency_mode=True),
                bank_acc=get_fake_acc(),
                bank_name=random.choice(c5_bank_names),
                phone=get_fake_phone(),
                money=get_fake_money()
            )
            penalty = random.choice(c5_penalties)
            sentence = f"{warn} {agency} thông báo: {acc}. Do đó, {cta}. {penalty}"
        else:
            # Giveaway Scam
            brand = random.choice(c5_promo_brands)
            reason = random.choice(c5_reasons)
            prize = random.choice(c5_prizes)
            action = random.choice(c5_scam_actions).format(
                fake_url=get_fake_url(agency_mode=False),
                money=get_fake_money()
            )
            prefix = random.choice(["Chúc mừng!", "THÔNG BÁO TRÚNG THƯỞNG:", "SỰ KIỆN LỚN:"])
            sentence = f"{prefix} {brand} {reason} dành tặng khách hàng phần quà gồm {prize}. Để nhận giải, {action} Số lượng có hạn!"
            
    if sentence and sentence not in generated_fake_set:
        generated_fake_set.add(sentence)
        counts[theme] += 1

print(f"Generated {len(generated_fake_set)} unique samples.")
print(f"Theme distribution: {counts}")

# Convert to list and trim to exact target
final_fake_list = list(generated_fake_set)[:target_fake_additional]

# Create final DataFrame
df_fake_additional = pd.DataFrame({
    "post_message": final_fake_list,
    "label": [1] * len(final_fake_list)
})

df_final_additional = pd.concat([df_real_additional, df_fake_additional], ignore_index=True)

# Shuffle the dataframe to mix real and fake samples nicely
df_final_additional = df_final_additional.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to additional_train.csv
print(f"Saving fully balanced additional_train.csv containing {len(df_final_additional)} rows...")
df_final_additional.to_csv(additional_csv_path, index=False)
print("File successfully saved!")

# Verify final distribution when merged with train.csv
total_real = train_real_count + len(df_final_additional[df_final_additional['label'] == 0])
total_fake = train_fake_count + len(df_final_additional[df_final_additional['label'] == 1])
print(f"Verification - Final Combined Dataset Distribution:")
print(f"--> Total REAL (0) = {total_real}")
print(f"--> Total FAKE (1) = {total_fake}")
print(f"--> Is perfectly balanced? {total_real == total_fake}")

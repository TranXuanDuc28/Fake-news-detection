import os
import csv
import random

# File paths
csv_path = r"d:\XuanDuc\TaiLieuKi8\CuoiKiHocSau\data\additional_train.csv"

# Lists for Group A: Government & Authority Impersonation
warning_words = [
    "CẢNH BÁO KHẨN CẤP TỪ COR QUAN CHỨC NĂNG:",
    "THÔNG BÁO QUAN TRỌNG:",
    "LỆNH TRIỆU TẬP KHẨN CẤP:",
    "QUYẾT ĐỊNH XỬ PHẠT HÀNH CHÍNH:",
    "LỆNH BẮT GIAM TẠM THỜI:",
    "CẢNH BÁO AN NINH MẠNG:",
    "YÊU CẦU PHỐI HỢP ĐIỀU TRA KHẨN CẤP:",
    "THÔNG BÁO KHÓA THUÊ BAO VĨNH VIỄN:",
    "LỆNH TRUY NÃ ĐẶC BIỆT THI HÀNH CƯỠNG CHẾ:",
    "THÔNG BÁO PHẠT NGUỘI GIAO THÔNG:"
]

agencies = [
    "Bộ Công an",
    "Viện Kiểm sát Nhân dân Tối cao",
    "Cơ quan Cảnh sát Điều tra Công an Thành phố Hà Nội",
    "Cơ quan Cảnh sát Điều tra Công an TP.HCM",
    "Cục Cảnh sát Giao thông",
    "Tổng cục Thuế Việt Nam",
    "Bảo hiểm Xã hội Việt Nam",
    "Tổng công ty Điện lực Việt Nam EVN",
    "Bộ Y tế",
    "Bộ Thông tin và Truyền thông",
    "Cục An ninh mạng và phòng chống tội phạm công nghệ cao",
    "Sở Giáo dục và Đào tạo",
    "Tổng công ty Cấp nước SAWACO",
    "Chi cục Thuế quận 1 TP.HCM",
    "Công an Quận Cầu Giấy Hà Nội"
]

accusations = [
    "tài khoản của ông/bà đang nằm trong danh sách đen nghi vấn rửa tiền và tài trợ khủng bố quốc tế",
    "số Căn cước công dân của bạn liên quan trực tiếp đến đường dây buôn lậu ma túy xuyên quốc gia",
    "phát hiện hành vi trốn thuế thu nhập cá nhân và sử dụng trái phép hóa đơn tài chính giả",
    "phương tiện giao thông của bạn đã vượt đèn đỏ và gây tai nạn nghiêm trọng rồi bỏ chạy tại giao lộ",
    "đang có hành vi vi phạm nghiêm trọng Luật An ninh mạng khi chia sẻ thông tin chống phá Nhà nước",
    "số điện thoại thuê bao của bạn phát tán hàng ngàn tin nhắn rác lừa đảo chiếm đoạt tài sản",
    "liên quan đến hồ sơ đen của vụ án chiếm đoạt tài sản quy mô lớn tại tập đoàn Vạn Thịnh Phát và SCB",
    "chưa hoàn thành nghĩa vụ đóng tiền điện nước kỳ gần nhất và đang có nợ đọng quá hạn kéo dài",
    "hệ thống phát hiện thông tin sinh trắc học và thẻ Căn cước công dân gắn chíp của bạn bị lỗi đồng bộ",
    "tài khoản ngân hàng của bạn đang giao dịch mờ ám với các trang web cá độ bóng đá và cờ bạc trực tuyến",
    "có dấu hiệu cấu kết với các đối tượng lừa đảo quốc tế để mở tài khoản thanh toán ảo nhằm rửa tiền sạch",
    "hồ sơ bảo hiểm xã hội của quý khách bị phát hiện khai man thông tin để trục lợi trợ cấp thất nghiệp"
]

bank_names = ["Vietcombank", "Techcombank", "BIDV", "Agribank", "Vietinbank", "MB Bank", "Sacombank", "VPBank", "TPBank", "ACB"]

call_to_actions = [
    "yêu cầu quý khách truy cập ngay cổng dịch vụ công trực tuyến tại {fake_url} để cập nhật thông tin phục vụ xác minh",
    "đề nghị tải và cài đặt ngay ứng dụng giám sát bảo mật 'Bộ Công An' tại đường link {fake_url} để quét khuôn mặt đối soát",
    "yêu cầu khẩn trương chuyển toàn bộ số dư tiền tiết kiệm hiện có sang tài khoản giám sát an toàn số {bank_acc} tại ngân hàng {bank_name} để kiểm toán tài chính",
    "yêu cầu nhấp vào liên kết {fake_url} để điền thông tin đăng nhập ngân hàng và mã xác thực OTP nhằm đối soát dòng tiền nghi vấn",
    "yêu cầu liên hệ khẩn cấp với Đại úy điều tra viên qua số hotline {phone} để làm việc gián tiếp và giữ bí mật thông tin",
    "vui lòng đóng phạt nguội số tiền {money} qua cổng dịch vụ trực tuyến tại link {fake_url} để hoàn thành nghĩa vụ thanh toán",
    "hãy truy cập trang liên kết chính phủ {fake_url} để điền tờ khai bảo mật và cung cấp mã OTP ngân hàng xác thực tài khoản",
    "yêu cầu đăng nhập vào hệ thống viễn thông quốc gia tại {fake_url} để làm thủ tục giải trình mở khóa thuê bao điện thoại"
]

penalties = [
    "Nếu trì hoãn hoặc bất hợp tác, cơ quan công an sẽ tiến hành phong tỏa tài khoản vĩnh viễn và khởi tố hình sự.",
    "Quá thời hạn 2 giờ kể từ khi nhận thông báo này, lệnh cưỡng chế và bắt tạm giam 3 tháng sẽ được thi hành.",
    "Mọi hành vi trốn tránh sẽ bị xử lý nghiêm khắc trước pháp luật và gửi lệnh triệu tập về nơi cư trú hoặc làm việc.",
    "Quá hạn nộp phạt, chúng tôi sẽ tiến hành tước quyền sử dụng giấy phép lái xe vĩnh viễn và niêm phong phương tiện vi phạm.",
    "Hết thời hạn quy định, chúng tôi sẽ tự động tạm ngắt dịch vụ cấp điện nước và thu hồi hợp đồng sử dụng đối với hộ gia đình.",
    "Nếu không xác minh kịp thời, toàn bộ tiền trong tài khoản của bạn sẽ bị đóng băng để phục vụ điều tra vô thời hạn."
]

# Lists for Group B: Giveaways, Brand Scams & Lucky Draws
promo_brands = [
    "Hãng hàng không quốc gia Vietnam Airlines",
    "Ứng dụng mua sắm trực tuyến Shopee Việt Nam",
    "Ví điện tử MoMo",
    "Tập đoàn Vingroup",
    "Tổng công ty Viễn thông Viettel",
    "Hệ thống siêu thị Thế Giới Di Động",
    "Mạng xã hội Facebook Việt Nam",
    "Ngân hàng Vietcombank",
    "Ngân hàng Techcombank",
    "Tập đoàn công nghệ Samsung Việt Nam",
    "Thương hiệu bia Heineken Việt Nam",
    "Công ty nước giải khát Coca-Cola",
    "Hãng hàng không Vietjet Air",
    "Tổng công ty Bưu điện Việt Nam VNPost",
    "Ứng dụng gọi xe Grab Việt Nam",
    "Hệ thống siêu thị điện máy xanh Coopmart",
    "Thương hiệu thời trang Adidas Việt Nam",
    "Ứng dụng thanh toán ZaloPay",
    "Ngân hàng BIDV Việt Nam",
    "Sàn thương mại điện tử Lazada"
]

reasons = [
    "nhân dịp kỷ niệm 30 năm ngày thành lập tập đoàn",
    "trong chương trình siêu tri ân khách hàng thân thiết năm 2026",
    "nhân ngày hội mua sắm khuyến mãi lớn nhất trong năm",
    "hệ thống đã lựa chọn ngẫu nhiên số thuê bao của bạn trúng giải đặc biệt",
    "nhân dịp đạt cột mốc 20 triệu người dùng hoạt động trên toàn quốc",
    "nhân dịp chào đón mùa hè rực rỡ và tri ân người tiêu dùng Việt Nam",
    "trong chiến dịch quảng bá sản phẩm mới và tặng quà trải nghiệm miễn phí"
]

prizes = [
    "1 chiếc xe máy Honda SH 150i trị giá 120 triệu đồng cùng 50 triệu đồng tiền mặt",
    "1 điện thoại thông minh iPhone 15 Pro Max trị giá 35 triệu đồng và voucher mua sắm 5 triệu",
    "phiếu mua hàng siêu thị trị giá 10 triệu đồng áp dụng mua sắm miễn phí toàn bộ sản phẩm",
    "gói quà tặng thanh toán hóa đơn đa năng trị giá 2.000.000đ cộng trực tiếp vào ví",
    "2 vé máy bay khứ hồi miễn phí đi bất kỳ chặng bay nội địa và quốc tế nào",
    "1 sổ tiết kiệm online trị giá 15.000.000đ với lãi suất cực kỳ ưu đãi",
    "bưu phẩm chứa tiền mặt 20 triệu đồng và nhiều quà tặng giá trị gửi từ nước ngoài chưa có người nhận",
    "gói data 4G tốc độ cao dung lượng không giới hạn sử dụng miễn phí trong vòng 1 năm",
    "phiếu quà tặng mua xe máy điện VinFast Feliz S trị giá 30 triệu đồng",
    "hộp quà bí ẩn chứa các thiết bị công nghệ Samsung trị giá lên tới 25 triệu đồng"
]

scam_actions = [
    "quý khách vui lòng nhấp vào đường link {fake_url} để điền thông tin cá nhân và đăng ký nhận quà giao tận nhà.",
    "truy cập ngay trang sự kiện chính thức {fake_url} để đăng nhập tài khoản ngân hàng trực tuyến nhận tiền mặt tức thì.",
    "yêu cầu thanh toán lệ phí làm hồ sơ nhận giải là {money} bằng cách nạp thẻ cào điện thoại qua liên kết {fake_url}.",
    "vui lòng click vào liên kết {fake_url} để xác thực số điện thoại và chia sẻ tin nhắn này đến 5 nhóm Facebook để hoàn tất thủ tục.",
    "truy cập ngay cổng vận chuyển {fake_url} để đóng phí lưu kho bưu phẩm là {money} và cập nhật địa chỉ giao nhận hàng chính xác.",
    "nhanh tay đăng ký tài khoản và điền mã OTP xác thực tại {fake_url} để kích hoạt gói quà tặng giới hạn này."
]

# Helper generators for fake details
def get_fake_url(agency_mode=True):
    sub = random.choice(["bocongan", "csgt", "gdt", "bhxh", "evn", "chuyentiengov", "bo-cong-an", "phatnguoi-csgt", "xacminh-taikhoan", "dichvucong-gov"]) if agency_mode else \
          random.choice(["shopee-trian", "vietnamairlines-gift", "momo-nhanloc", "viettel-trungthuong", "vinfast-feliz", "thegioididong-quatang", "facebook-nhangiai", "vcb-digibank", "samsung-s24", "heineken-he", "grab-khuyenmai"])
    domain = random.choice([".com", ".net", ".org", ".cc", ".xyz", ".info", ".net.vn", ".co", ".online", ".club", ".vip"])
    path = random.choice(["/login", "/auth", "/trian", "/nhangiai", "/xacminh", "/home", "/service", "/register"])
    return f"http://{sub}{domain}{path}"

def get_fake_phone():
    prefix = random.choice(["09", "08", "07", "03", "05"])
    return prefix + "".join(str(random.randint(0, 9)) for _ in range(8))

def get_fake_acc():
    return "".join(str(random.randint(0, 9)) for _ in range(random.choice([10, 12, 13])))

def get_fake_money():
    return f"{random.choice([50, 100, 200, 500, 800, 1000, 1500])}.000đ"

# Generate dataset
generated_set = set()
target_count = 1500  # Generate 1500 total samples (750 authority, 750 giveaway)

# Generate Group A (Authority)
while len(generated_set) < target_count // 2:
    warn = random.choice(warning_words)
    agency = random.choice(agencies)
    acc = random.choice(accusations)
    cta = random.choice(call_to_actions).format(
        fake_url=get_fake_url(agency_mode=True),
        bank_acc=get_fake_acc(),
        bank_name=random.choice(bank_names),
        phone=get_fake_phone(),
        money=get_fake_money()
    )
    penalty = random.choice(penalties)
    
    # Construct sentence
    sentence = f"{warn} {agency} thông báo: {acc}. Do đó, {cta}. {penalty}"
    generated_set.add(sentence)

# Generate Group B (Giveaways & Scams)
while len(generated_set) < target_count:
    brand = random.choice(promo_brands)
    reason = random.choice(reasons)
    prize = random.choice(prizes)
    action = random.choice(scam_actions).format(
        fake_url=get_fake_url(agency_mode=False),
        money=get_fake_money()
    )
    
    # Construct sentence
    prefix = random.choice(["Chúc mừng!", "TIN KHẨN CẤP:", "THÔNG BÁO TRÚNG THƯỞNG:", "SỰ KIỆN LỚN:", "Cơ hội có một không hai!"])
    sentence = f"{prefix} {brand} {reason} dành tặng khách hàng phần quà gồm {prize}. Để nhận giải, {action} Số lượng có hạn!"
    generated_set.add(sentence)

# Append to additional_train.csv
print(f"Adding {len(generated_set)} unique synthetic scam/impersonation samples to {csv_path}...")

with open(csv_path, mode='a', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    for text in generated_set:
        writer.writerow([text, 1])

print("Completed appending synthetic data!")

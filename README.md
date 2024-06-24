# Survey Reinforcement Learning in Bomberman
Đây là repository chứa mã nguồn và tài liệu liên quan đến đồ án môn Trí tuệ nhân tạo - CS106.O21.KHTN, nghiên cứu về reinforcement learning trong Bomberman.

---
### Giảng viên hướng dẫn
TS. Lương Ngọc Hoàng

---
### Thành viên nhóm
- 22520004 - Trần Như Cẩm Nguyên
- 22520593 - Nguyễn Thanh Hỷ
- 22520946 - Lê Tín Nghĩa

---
### Giới thiệu
Người chơi di chuyển Bomberman trong mê cung, đặt bom để phá hủy các khối gỗ cũng như tiêu diệt kẻ địch. Mục tiêu là tiêu diệt tất cả cả địch và thu thập vật phẩm.

<img width="269" alt="image" src="https://github.com/nguyenthanhhy0108/CS106-RL-in-Bomberman/assets/73975520/888810b4-bd7c-4244-9e18-dcde4b5193bb">

---
## Cài đặt

### Clone repository
```
git clone https://github.com/nguyenthanhhy0108/CS106-RL-in-Bomberman
```

### Người dùng điều khiển để chơi
```
python main.py play --my-agent user_agent
```

### Quan sát máy tự chơi
```
python main.py play
```

### Train
Ví dụ là `sac_agent`. Các agent khác xem trong folder `agent_code`.
```
python main.py play --my-agent sac_agent --train 1 --n-rounds 1000
```

### Chạy với mô hình đã train
```
python main.py play --my-agent sac_agent
```

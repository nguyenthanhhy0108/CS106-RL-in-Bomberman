# Survey Reinforcement Learning in Bomberman

Đây là repository chứa mã nguồn và tài liệu liên quan đến đồ án môn Trí tuệ nhân tạo - CS106.O21.KHTN, nghiên cứu về reinforcement learning trong Bomberman.

## Giảng viên hướng dẫn
TS. Lương Ngọc Hoàng

## Thành viên nhóm
- 22520004 - Trần Như Cẩm Nguyên
- 22520593 - Nguyễn Thanh Hỷ
- 22520946 - Lê Tín Nghĩa

## Cấu trúc thư mục

- agent_code/: Code của các agent trong game.
- assets/: Tài nguyên đồ họa cho Bomberman.
- logs/: Thư mục chứa các file log của game.
- replays/: Các file ghi lại các trận đấu replay của game.
- results/: Kết quả thực nghiệm và đánh giá các mô hình học tăng cường.
- screenshots/: Ảnh chụp màn hình của game hoặc kết quả thực nghiệm.
- Slides.pdf: Bài trình bày Slide thuyết trình dự án.
- agents.py, environment.py, events.py, fallbacks.py, items.py, main.py, replay.py, settings.py: Các file code chính của dự án.
  
## Cài đặt

- Đầu tiên, cần clone repository
```
git clone https://github.com/nguyenthanhhy0108/CS106-RL-in-Bomberman
```

### Tự điều khiển để chơi
```
python main.py play --my-agent user_agent
```

### Quan sát máy tự chơi
```
python main.py play
```

### Train
```
python main.py play --my-agent tpl_agent --train 1 --n-rounds 1000
```

### Cho mô hình đã train chơi
```
python main.py play --my-agent tpl_agent
```

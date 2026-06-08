import torch

# 1. Chuẩn bị dữ liệu (3 chuyến taxi)
# x = 1, 2, 3. Sự thật y = 5, 7, 9 (vì y = 2x + 3)
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([5.0, 7.0, 9.0])

# 2. Khởi tạo Thông số (Đoán bừa w=0, b=0)
# CẢ HAI đều cần phải học, nên CẢ HAI đều phải BẬT GHI CHÉP!
w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

learning_rate = 0.05

print("BẮT ĐẦU HUẤN LUYỆN...\n")

# 3. Vòng lặp huấn luyện (Training Loop)
for epoch in range(500): # Cho máy học 500 vòng
    
    # Bước A: Forward Pass
    y_pred = (w * x) + b
    
    # Bước B: Tính Loss (Dùng hàm Trung bình Bình phương Sai số - MSE)
    loss = torch.mean((y_pred - y_true)**2)
    
    # Bước C: Tính Gradient (Kỳ quan của Autograd)
    loss.backward()
    
    # Bước D: Update Weights (Gradient Descent)
    with torch.no_grad(): # Tắt ghi chép khi đang thao tác vật lý
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # Bước E: Xóa bảng
        w.grad.zero_()
        b.grad.zero_()
        
    # In kết quả sau mỗi 100 vòng để quan sát
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | w: {w.item():.4f}, b: {b.item():.4f}")

print(f"\n=> KẾT QUẢ CUỐI CÙNG: Phương trình máy tìm được là y = {w.item():.2f}x + {b.item():.2f}")
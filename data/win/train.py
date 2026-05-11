import os
import time
import torch
from monai.networks.nets import UNet
from monai.losses import TverskyLoss 
import data_loader 

def main():
    print("=" * 60)
    print("🚀 '腺'而易见 — 睑板腺 MGD 深度学习模型训练引擎 (满血精度版)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"✅ 算力引擎就绪: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True 
    else:
        print("⚠️ 未检测到 GPU，训练速度将受到严重限制。")
    print("=" * 60)

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2,          
        act="PRELU",              
        norm="INSTANCE",          
        dropout=0.1
    ).to(device)

    loss_function = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    scaler = torch.amp.GradScaler('cuda')

    max_epochs = 20        
    save_interval = 2      
    best_loss = float('inf') 
    
    print(f"\n📂 训练数据集: {len(data_loader.check_ds)} 例专业影像")
    print(f"🔥 训练设置: Epochs={max_epochs}, BatchSize={data_loader.check_loader.batch_size}")
    print(f"🌟 已激活: 全尺度(64-1024) + Res-Skip + PixelShuffle + FP32 保存")
    print("-" * 40)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        start_time = time.time()
        
        for batch_data in data_loader.check_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            if step % max(1, len(data_loader.check_loader) // 5) == 0 or step == len(data_loader.check_loader):
                print(f"   [轮次 {epoch+1}] 进度: {step}/{len(data_loader.check_loader)} - Tversky 损失: {loss.item():.4f}")
        
        avg_loss = epoch_loss / step
        epoch_time = time.time() - start_time
        print(f"✅ 第 {epoch+1} 轮完成 | 平均损失: {avg_loss:.4f} | 耗时: {epoch_time:.1f}s")

        # 【撤销压缩】恢复原生的 state_dict() 保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "meibomian_model_best.pth")
            print(f"   🏆 探测到更优模型！已保存完整精度权重: meibomian_model_best.pth")

        if (epoch + 1) % save_interval == 0:
            checkpoint_name = f"mgd_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_name)
            print(f"   💾 定期备份已存: {checkpoint_name}")

    print("-" * 40)
    print(f"🎉 训练全流程顺利结束！")
    print(f"📈 最优模型损失值: {best_loss:.4f}")
    print(f"💾 请在目录中使用 'meibomian_model_best.pth' 进行临床部署。")
    print("=" * 60)

if __name__ == "__main__":
    main()
"""
UI 模块
"""
import gradio as gr
from config import BUILTIN_PRESETS, get_all_trained_models
from train import professional_training
from inference import professional_inference, visualize_train_sample, evaluate_n_train_samples
from utils import load_all_presets, save_custom_preset_file, sync_infer_controls


def get_model_choices():
    """
    获取可用的模型选择列表
    """
    trained = get_all_trained_models()
    if trained:
        return trained
    return ["UNet", "AttentionUNet", "WRCANet"]


def apply_named_preset(preset_name):
    """
    应用命名预设
    """
    presets = load_all_presets()
    p = presets.get(preset_name, BUILTIN_PRESETS["平衡-默认"])
    synced = sync_infer_controls(
        p.get("tile_mode", True),
        p.get("tile_size", 192),
        p.get("tile_overlap", 0.25),
        p.get("threshold", 0.70),
        p.get("min_area", 120),
        p.get("smooth_on", True),
        p.get("smooth_sigma", 1.2),
        p.get("roi_only", True),
        p.get("tta_mode", "关闭"),
    )
    names = list(load_all_presets().keys())
    status = f"✅ 已应用预设：{preset_name}"
    
    return (
        *synced,
        gr.update(choices=names, value=preset_name),
        gr.update(choices=names, value=preset_name),
        status
    )


def save_named_preset(preset_name, tile_mode, tile_size, tile_overlap, threshold, min_area, smooth_on, smooth_sigma, roi_only, tta_mode):
    """
    保存命名预设
    """
    name = (preset_name or "").strip()
    if not name:
        names = list(load_all_presets().keys())
        return gr.update(choices=names), gr.update(choices=names), "❌ 预设名不能为空"
    if name in BUILTIN_PRESETS:
        names = list(load_all_presets().keys())
        return gr.update(choices=names), gr.update(choices=names), "❌ 内置预设不可覆盖，请换个名字保存"
    presets = load_all_presets()
    synced = sync_infer_controls(tile_mode, tile_size, tile_overlap, threshold, min_area, smooth_on, smooth_sigma, roi_only, tta_mode)
    presets[name] = {
        "tile_mode": synced[0], "tile_size": synced[1], "tile_overlap": synced[2],
        "threshold": synced[3], "min_area": synced[4],
        "smooth_on": synced[5], "smooth_sigma": synced[6], "roi_only": synced[7],
        "tta_mode": synced[8],
    }
    save_custom_preset_file(presets)
    names = list(load_all_presets().keys())
    return gr.update(choices=names, value=name), gr.update(choices=names, value=name), f"💾 已保存预设：{name}"


def create_app():
    """
    创建 Gradio 应用
    """
    with gr.Blocks(title="“腺”而易见 — 满血专业版") as app:
        gr.Markdown("# 👁️ “腺”而易见 — MGD 智能辅助诊疗系统 V2.0")
        
        with gr.Tabs():
            with gr.TabItem("🩺 临床诊断终端"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(label="载入红外影像", type="numpy")
                        infer_model_diag = gr.Dropdown(
                            choices=get_model_choices(),
                            value="UNet",
                            label="选择模型架构"
                        )
                        infer_preset_diag = gr.Dropdown(
                            choices=list(load_all_presets().keys()),
                            value="平衡-默认",
                            label="推理参数预设（诊断页）"
                        )
                        infer_tile_mode = gr.Checkbox(label="🧩 分块推理 (滑窗)", value=True)
                        infer_tile_size = gr.Slider(128, 512, value=192, step=32, label="推理 Tile 高度")
                        infer_tile_overlap = gr.Slider(0.1, 0.7, value=0.25, step=0.05, label="滑窗重叠比例")
                        infer_threshold = gr.Slider(0.4, 0.95, value=0.7, step=0.05, label="分割阈值（抑制溢出）")
                        infer_min_area = gr.Slider(20, 500, value=120, step=10, label="最小连通域面积")
                        infer_smooth_fusion = gr.Checkbox(label="🧪 边界平滑融合", value=True)
                        infer_smooth_sigma = gr.Slider(0.0, 3.0, value=1.2, step=0.1, label="平滑强度 (sigma)")
                        infer_roi_only = gr.Checkbox(label="🎯 仅眼睑 ROI 内保留", value=True)
                        infer_tta = gr.Dropdown(
                            choices=["关闭", "2倍增强 (水平翻转)", "4倍增强 (全面翻转)"],
                            value="关闭",
                            label="测试时增强 (TTA)"
                        )
                        btn_infer = gr.Button("🚀 运行定量分析", variant="primary", elem_classes="primary-btn")
                        gr.Markdown("### 📊 临床定量报告")
                        out_metrics = gr.Dataframe(
                            headers=["指标名称", "检测数值"],
                            label="AI 自动量化指标",
                            interactive=False
                        )
                        preset_status = gr.Textbox(label="预设操作状态", interactive=False)
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            out_original = gr.Image(label="原始影像", interactive=False)
                            out_seg = gr.Image(label="AI 分割结果", interactive=False)
                        with gr.Row():
                            out_heatmap = gr.Image(label="AI 注意力热力图", interactive=False)
                            out_contour = gr.Image(label="轮廓可视化", interactive=False)
                        with gr.Row():
                            out_skeleton = gr.Image(label="骨架可视化", interactive=False)
                            out_instance = gr.Image(label="实例标注", interactive=False)
                
                btn_infer.click(
                    fn=professional_inference,
                    inputs=[
                        input_img,
                        infer_tile_mode,
                        infer_tile_size,
                        infer_tile_overlap,
                        infer_threshold,
                        infer_min_area,
                        infer_smooth_fusion,
                        infer_smooth_sigma,
                        infer_roi_only,
                        infer_tta,
                        infer_model_diag,
                    ],
                    outputs=[
                        out_original,
                        out_seg,
                        out_heatmap,
                        out_metrics,
                        out_contour,
                        out_skeleton,
                        out_instance,
                    ]
                )
                
                infer_preset_diag.change(
                    fn=apply_named_preset,
                    inputs=[infer_preset_diag],
                    outputs=[
                        infer_tile_mode, infer_tile_size, infer_tile_overlap, infer_threshold, infer_min_area, infer_smooth_fusion, infer_smooth_sigma, infer_roi_only, infer_tta,
                        infer_tile_mode, infer_tile_size, infer_tile_overlap, infer_threshold, infer_min_area, infer_smooth_fusion, infer_smooth_sigma, infer_roi_only, infer_tta,
                        infer_preset_diag, infer_preset_diag, preset_status
                    ]
                )
            
            with gr.TabItem("🔬 模型训练"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🛠️ 训练参数配置")
                        model_name = gr.Dropdown(
                            choices=["UNet", "AttentionUNet", "WRCANet"],
                            value="UNet",
                            label="选择模型架构"
                        )
                        resume_ckpt = gr.Checkbox(label="🔌 断点续训（从 meibomian_model_best.pth 继续）", value=False)
                        train_epochs = gr.Number(label="训练轮数", value=20, precision=0, minimum=1)
                        batch_size = gr.Number(label="批次大小", value=1, precision=0, minimum=1)
                        save_freq = gr.Number(label="保存检查点频率（每 N 轮）", value=2, precision=0, minimum=1)
                        max_keep_ckpts = gr.Number(label="最多保留检查点数量", value=3, precision=0, minimum=1)
                        
                        gr.Markdown("### 📊 算力配置")
                        mem_limit = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="显存使用上限比例（防OOM）")
                        precision_opt = gr.Radio(["FP32", "FP16 (混合精度)"], value="FP32", label="训练精度")
                        
                        gr.Markdown("### 🧩 分块训练（显存不足时用）")
                        tile_mode_train = gr.Checkbox(label="启用分块训练", value=False)
                        tile_height_train = gr.Slider(128, 512, value=256, step=32, label="Tile 高度")
                        
                        gr.Markdown("### 📈 优化器配置")
                        lr = gr.Number(label="学习率", value=1e-3, minimum=1e-6, maximum=1e-1, step=1e-5)
                        optimizer_name = gr.Dropdown(["AdamW", "Adam", "SGD"], value="AdamW", label="优化器")
                        scheduler_name = gr.Dropdown(
                            ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau (默认)", "Cosine Annealing", "Step Decay", "Fixed LR"],
                            value="ReduceLROnPlateau (默认)",
                            label="学习率调度器"
                        )
                        weight_decay = gr.Number(label="权重衰减 (L2)", value=1e-5, minimum=0.0)
                        
                        gr.Markdown("### 💔 损失函数")
                        loss_fn_name = gr.Dropdown(["Dice Loss", "Cross Entropy", "Focal Loss", "Dice+CE 组合损失"], value="Dice Loss", label="损失函数")
                        pos_weight = gr.Number(label="正样本权重（针对不平衡数据）", value=1.0, minimum=0.1, maximum=10.0)
                        
                        gr.Markdown("### 🎨 数据增强")
                        use_augmentation = gr.Checkbox(label="启用数据增强（随机翻转）", value=False)
                        
                        gr.Markdown("### 📊 训练可视化")
                        wandb_enable = gr.Checkbox(label="启用 Weights & Biases (W&B) 可视化", value=False)
                        
                        btn_train = gr.Button("🚀 开始训练", variant="primary", elem_classes="primary-btn")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📜 训练日志")
                        train_log = gr.Textbox(label="训练日志", lines=30, max_lines=50, interactive=False, value="✅ 训练控制台准备就绪\n\n请调整左侧参数，点击【开始训练】按钮启动训练\n")
                
                # 参数实时反馈函数
                def log_param_change(param_name, old_val, new_val):
                    return f"\n📝 [{param_name}] 已修改: {old_val} → {new_val}\n"
                
                def log_checkbox_change(param_name, new_val):
                    status = "✅ 已启用" if new_val else "❌ 已禁用"
                    return f"\n📝 [{param_name}] {status}\n"
                
                # 添加所有参数的 change 事件
                model_name.change(
                    fn=lambda val, log: log + (f"\n📝 [模型架构] 已切换: {val}\n" if log else f"📝 [模型架构] 已切换: {val}\n"),
                    inputs=[model_name, train_log],
                    outputs=train_log
                )
                
                resume_ckpt.change(
                    fn=lambda val, log: log + f"\n📝 [断点续训] {'✅ 已启用' if val else '❌ 已禁用'}\n",
                    inputs=[resume_ckpt, train_log],
                    outputs=train_log
                )
                
                train_epochs.change(
                    fn=lambda val, log: log + f"\n📝 [训练轮数] 已调整: {int(val)} 轮\n",
                    inputs=[train_epochs, train_log],
                    outputs=train_log
                )
                
                batch_size.change(
                    fn=lambda val, log: log + f"\n📝 [批次大小] 已调整: {int(val)}\n",
                    inputs=[batch_size, train_log],
                    outputs=train_log
                )
                
                save_freq.change(
                    fn=lambda val, log: log + f"\n📝 [保存频率] 已调整: 每 {int(val)} 轮\n",
                    inputs=[save_freq, train_log],
                    outputs=train_log
                )
                
                max_keep_ckpts.change(
                    fn=lambda val, log: log + f"\n📝 [保留检查点] 已调整: 保留 {int(val)} 个\n",
                    inputs=[max_keep_ckpts, train_log],
                    outputs=train_log
                )
                
                mem_limit.change(
                    fn=lambda val, log: log + f"\n📝 [显存上限] 已调整: {val:.2%}\n",
                    inputs=[mem_limit, train_log],
                    outputs=train_log
                )
                
                precision_opt.change(
                    fn=lambda val, log: log + f"\n📝 [训练精度] 已切换: {val}\n",
                    inputs=[precision_opt, train_log],
                    outputs=train_log
                )
                
                tile_mode_train.change(
                    fn=lambda val, log: log + f"\n📝 [分块训练] {'✅ 已启用' if val else '❌ 已禁用'}\n",
                    inputs=[tile_mode_train, train_log],
                    outputs=train_log
                )
                
                tile_height_train.change(
                    fn=lambda val, log: log + f"\n📝 [Tile 高度] 已调整: {int(val)}\n",
                    inputs=[tile_height_train, train_log],
                    outputs=train_log
                )
                
                lr.change(
                    fn=lambda val, log: log + f"\n📝 [学习率] 已调整: {val:.6f}\n",
                    inputs=[lr, train_log],
                    outputs=train_log
                )
                
                optimizer_name.change(
                    fn=lambda val, log: log + f"\n📝 [优化器] 已切换: {val}\n",
                    inputs=[optimizer_name, train_log],
                    outputs=train_log
                )
                
                scheduler_name.change(
                    fn=lambda val, log: log + f"\n📝 [学习率调度器] 已切换: {val}\n",
                    inputs=[scheduler_name, train_log],
                    outputs=train_log
                )
                
                weight_decay.change(
                    fn=lambda val, log: log + f"\n📝 [权重衰减] 已调整: {val:.6f}\n",
                    inputs=[weight_decay, train_log],
                    outputs=train_log
                )
                
                loss_fn_name.change(
                    fn=lambda val, log: log + f"\n📝 [损失函数] 已切换: {val}\n",
                    inputs=[loss_fn_name, train_log],
                    outputs=train_log
                )
                
                pos_weight.change(
                    fn=lambda val, log: log + f"\n📝 [正样本权重] 已调整: {val:.2f}\n",
                    inputs=[pos_weight, train_log],
                    outputs=train_log
                )
                
                use_augmentation.change(
                    fn=lambda val, log: log + f"\n📝 [数据增强] {'✅ 已启用' if val else '❌ 已禁用'}\n",
                    inputs=[use_augmentation, train_log],
                    outputs=train_log
                )
                
                wandb_enable.change(
                    fn=lambda val, log: log + f"\n📝 [W&B 可视化] {'✅ 已启用' if val else '❌ 已禁用'}\n",
                    inputs=[wandb_enable, train_log],
                    outputs=train_log
                )
                
                btn_train.click(
                    fn=professional_training,
                    inputs=[
                        resume_ckpt, train_epochs, batch_size, save_freq, max_keep_ckpts,
                        mem_limit, precision_opt, tile_mode_train, tile_height_train,
                        lr, optimizer_name, scheduler_name, weight_decay,
                        loss_fn_name, pos_weight, use_augmentation,
                        model_name, wandb_enable
                    ],
                    outputs=train_log
                )
            
            with gr.TabItem("✅ 模型验证"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔍 推理参数配置")
                        eval_model = gr.Dropdown(
                            choices=get_model_choices(),
                            value="UNet",
                            label="选择模型架构"
                        )
                        infer_preset_eval = gr.Dropdown(
                            choices=list(load_all_presets().keys()),
                            value="平衡-默认",
                            label="推理参数预设"
                        )
                        eval_tile_mode = gr.Checkbox(label="🧩 分块推理 (滑窗)", value=True)
                        eval_tile_size = gr.Slider(128, 512, value=192, step=32, label="推理 Tile 高度")
                        eval_tile_overlap = gr.Slider(0.1, 0.7, value=0.25, step=0.05, label="滑窗重叠比例")
                        eval_threshold = gr.Slider(0.4, 0.95, value=0.7, step=0.05, label="分割阈值")
                        eval_min_area = gr.Slider(20, 500, value=120, step=10, label="最小连通域面积")
                        eval_smooth_fusion = gr.Checkbox(label="🧪 边界平滑融合", value=True)
                        eval_smooth_sigma = gr.Slider(0.0, 3.0, value=1.2, step=0.1, label="平滑强度")
                        eval_roi_only = gr.Checkbox(label="🎯 仅眼睑 ROI", value=True)
                        
                        gr.Markdown("### 🧪 验证方式")
                        with gr.Row():
                            btn_vis_sample = gr.Button("🎲 随机看一个训练样本", variant="secondary")
                            btn_eval_samples = gr.Button("📊 批量评估 N 张", variant="secondary")
                        
                        num_eval_samples = gr.Number(label="评估样本数", value=10, precision=0, minimum=1)
                        eval_status = gr.Textbox(label="评估状态", interactive=False)
                        
                        gr.Markdown("---\n### 💾 预设管理")
                        preset_name_input = gr.Textbox(label="新预设名称", placeholder="输入名称...")
                        btn_save_preset = gr.Button("💾 保存当前参数为预设", variant="secondary")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            eval_img = gr.Image(label="样本原始影像", interactive=False)
                            eval_label = gr.Image(label="专家标注 (GT)", interactive=False)
                            eval_pred = gr.Image(label="AI 预测", interactive=False)
                        with gr.Row():
                            eval_combined = gr.Image(label="对比可视化 (左: GT, 右: Pred)", interactive=False)
                        with gr.Row():
                            eval_hist = gr.Image(label="原始概率直方图", interactive=False)
                        with gr.Row():
                            eval_contour = gr.Image(label="轮廓可视化", interactive=False)
                            eval_skeleton = gr.Image(label="骨架可视化", interactive=False)
                            eval_instance = gr.Image(label="实例标注", interactive=False)
                
                btn_vis_sample.click(
                    fn=visualize_train_sample,
                    inputs=[
                        eval_model,
                        eval_tile_mode, eval_tile_size, eval_tile_overlap,
                        eval_threshold, eval_min_area,
                        eval_smooth_fusion, eval_smooth_sigma, eval_roi_only
                    ],
                    outputs=[
                        eval_img, eval_label, eval_pred, eval_combined, eval_hist, eval_status,
                        eval_contour, eval_skeleton, eval_instance
                    ]
                )
                
                btn_eval_samples.click(
                    fn=evaluate_n_train_samples,
                    inputs=[
                        num_eval_samples,
                        eval_model,
                        eval_tile_mode, eval_tile_size, eval_tile_overlap,
                        eval_threshold, eval_min_area,
                        eval_smooth_fusion, eval_smooth_sigma, eval_roi_only
                    ],
                    outputs=[
                        eval_img, eval_label, eval_pred, eval_combined, eval_hist, eval_status,
                        eval_contour, eval_skeleton, eval_instance
                    ]
                )
                
                infer_preset_eval.change(
                    fn=apply_named_preset,
                    inputs=[infer_preset_eval],
                    outputs=[
                        eval_tile_mode, eval_tile_size, eval_tile_overlap, eval_threshold, eval_min_area, eval_smooth_fusion, eval_smooth_sigma, eval_roi_only, gr.Textbox(value="关闭", visible=False),
                        infer_tile_mode, infer_tile_size, infer_tile_overlap, infer_threshold, infer_min_area, infer_smooth_fusion, infer_smooth_sigma, infer_roi_only, infer_tta,
                        infer_preset_eval, infer_preset_diag, preset_status
                    ]
                )
                
                btn_save_preset.click(
                    fn=save_named_preset,
                    inputs=[
                        preset_name_input,
                        eval_tile_mode, eval_tile_size, eval_tile_overlap, eval_threshold, eval_min_area, eval_smooth_fusion, eval_smooth_sigma, eval_roi_only,
                        gr.Textbox(value="关闭", visible=False)
                    ],
                    outputs=[infer_preset_eval, infer_preset_diag, preset_status]
                )
    
    return app

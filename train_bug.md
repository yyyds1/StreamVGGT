- 严重: StreamVGGT 的输入图像形状和 ActionVGGT 期望形状不一致，实际会把每帧图像“偷偷扩成更多帧”送进 patch embed，显存会被直接放大。
在数据集里，多相机图像被沿高度拼接成了 [F, C, 3*H, W]，见 lerobot_latent_dataset.py (line 354) 和 lerobot_latent_dataset.py (line 399)。
但模型是按默认 ActionVGGT() 初始化的，内部固定 img_height=518, img_width=518，见 train_va.py (line 84)。
到 aggregator 里又直接 images.reshape(-1, C, self.image_height, self.image_width)，见 aggregator.py (line 268)。如果输入真实高度是 3*518，这里会把 batch/frame 维隐式扩成 3 倍，不只是错，显存和 token 数也会一起炸。
- 严重: StreamVGGT 相比 va 确实少了最关键的显存优化，va 有 activation checkpointing，StreamVGGT 没真正启用。
va 在训练前明确调用了 apply_ac(self.transformer)，见 train.py (line 90)。
StreamVGGT 导入了 apply_ac，但没调用，只是走了一个 config.gradient_checkpointing 分支，见 train_va.py (line 114)。
而我检索了 actionvggt/rdt，并没有 gradient_checkpointing_enable() 实现，所以这个开关现在基本是假的。再加上默认配置里它还是 False，见 va_robotwin_train_rdt_cfg.py (line 28)。
- 严重: README 默认跑到的不是轻量的 va_robotwin_train_cfg，而是更重的 robotwin_train_rdt_cfg。
配置注册里 robotwin_train 指向了 va_robotwin_train_rdt_cfg，见 init.py (line 6)。
这个配置走的是原图 518x518 + window_size=4 + chunk_size=24 + ActionVGGT + RDT 双模型，见 va_robotwin_train_rdt_cfg.py (line 23)。
va 那边则是 latent 路径，本身就轻得多，不是同一个显存级别。
- 高: StreamVGGT 训练代码里还有几处明显写错，说明这条后训练路径本身还没对齐完，不只是 OOM。
compute_loss() 访问了不存在的 input_dict['latent_dict'] 和 input_dict['action_dict']['targets']，见 train_va.py (line 422)。
但 _prepare_input_dict() 实际只构造了 image_dict、action_dict、pred_action_chunk_dict，见 train_va.py (line 395)。
另外 compute_loss() 返回的是按帧向量，没有 .mean()，loss.backward() 对非标量也会出错，见 train_va.py (line 438) 和 train_va.py (line 534)。
- 中: 训练循环还有别的逻辑错误，不像内存泄漏，但说明代码路径没跑通。
accumulated_latent_losses 从头到尾没 append，却在日志阶段 torch.stack(accumulated_latent_losses)，见 train_va.py (line 499) 和 train_va.py (line 550)。
action_chunk_end_idx = min(action_chunk_start_idx + chunk_size, F) 这里切的是 flatten 后的 action 序列，却拿 frame 数 F 截断，见 train_va.py (line 381)。这更像索引 bug，不是泄漏。
- 中: 我没看到典型“显存持续增长型”的泄漏写法。
训练里 append 进列表的 loss 做了 .detach()，见 train_va.py (line 538)。
_add_noise 和 _prepare_input_dict 也都包了 @torch.no_grad()，见 train_va.py (line 252) 和 train_va.py (line 305)。
所以更像是“模型/输入规模过大 + 少了 checkpointing + 形状 bug”，不是经典 Python 容器持图泄漏。

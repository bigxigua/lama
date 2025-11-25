#!/usr/bin/env python3
"""
视频去字幕脚本
使用LaMa预训练模型对视频进行去字幕处理
"""

import os
import sys
import logging
import argparse
import cv2
import numpy as np
import torch
import yaml
import subprocess
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from contextlib import contextmanager
from io import StringIO

# 设置环境变量（CPU优化：使用多线程）
# 注意：如果使用GPU，这些设置会被GPU计算覆盖
cpu_threads = os.environ.get("CPU_THREADS", "16")  # 默认使用16核
os.environ["OMP_NUM_THREADS"] = cpu_threads
os.environ["OPENBLAS_NUM_THREADS"] = cpu_threads
os.environ["MKL_NUM_THREADS"] = cpu_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = cpu_threads
os.environ["NUMEXPR_NUM_THREADS"] = cpu_threads

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.data import pad_img_to_modulo, ceil_modulo
from saicinpainting.training.trainers import load_checkpoint

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@contextmanager
def suppress_stdout():
    """临时抑制stdout和logging输出的上下文管理器（保留stderr用于错误信息）"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        # 保存当前所有logger的级别
        old_levels = {}
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            old_levels[id(handler)] = handler.level
            handler.setLevel(logging.ERROR)  # 只显示ERROR级别以上的日志

        # 也设置其他可能相关的logger
        for logger_name in ["pytorch_lightning", "lightning", "saicinpainting"]:
            logger = logging.getLogger(logger_name)
            old_levels[logger_name] = logger.level
            logger.setLevel(logging.ERROR)

        try:
            sys.stdout = devnull
            yield
        finally:
            sys.stdout = old_stdout
            # 恢复logger级别
            for handler in root_logger.handlers:
                if id(handler) in old_levels:
                    handler.setLevel(old_levels[id(handler)])
            for logger_name, level in old_levels.items():
                if isinstance(logger_name, str):
                    logging.getLogger(logger_name).setLevel(level)


def create_default_config():
    """创建默认的模型配置"""
    config = {
        "run_title": "",
        "training_model": {
            "kind": "default",
            "visualize_each_iters": 1000,
            "concat_mask": True,
            "store_discr_outputs_for_vis": True,
            "predict_only": True,
        },
        "losses": {
            "l1": {"weight_missing": 0, "weight_known": 10},
            "perceptual": {"weight": 0},
            "adversarial": {
                "kind": "r1",
                "weight": 10,
                "gp_coef": 0.001,
                "mask_as_fake_target": True,
                "allow_scale_mask": True,
            },
            "feature_matching": {"weight": 100},
            "resnet_pl": {
                "weight": 30,
                "weights_path": os.environ.get("TORCH_HOME", "."),
            },
        },
        "generator": {
            "kind": "ffc_resnet",
            "input_nc": 4,
            "output_nc": 3,
            "ngf": 64,
            "n_downsampling": 3,
            "n_blocks": 18,
            "add_out_act": "sigmoid",
            "init_conv_kwargs": {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
            "downsample_conv_kwargs": {
                "ratio_gin": 0,
                "ratio_gout": 0,
                "enable_lfu": False,
            },
            "resnet_conv_kwargs": {
                "ratio_gin": 0.75,
                "ratio_gout": 0.75,
                "enable_lfu": False,
            },
        },
        "visualizer": {"kind": "noop"},
    }
    return OmegaConf.create(config)


def detect_subtitle_region(
    frame,
    x=0.00065359477124183,
    y=0.7529411764705882,
    width=0.9952069716775599,
    height=0.05294117647058831,
    detect_white_text=True,
    white_threshold=200,
):
    """
    根据已知坐标创建字幕区域mask，可选地只检测白色文字

    Args:
        frame: 输入帧 (H, W, 3) BGR格式
        x: 字幕区域左上角x坐标（归一化，0-1）
        y: 字幕区域左上角y坐标（归一化，0-1）
        width: 字幕区域宽度（归一化，0-1）
        height: 字幕区域高度（归一化，0-1）
        detect_white_text: 是否只检测白色文字，默认True
        white_threshold: 白色阈值（0-255），RGB/BGR值都大于此值认为是白色，默认200

    Returns:
        mask: 字幕区域的mask (H, W)，255表示字幕区域，0表示其他区域
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 将归一化坐标转换为像素坐标
    x_pixel = int(x * w)
    y_pixel = int(y * h)
    width_pixel = int(width * w)
    height_pixel = int(height * h)

    # 确保坐标在有效范围内
    x_pixel = max(0, min(x_pixel, w - 1))
    y_pixel = max(0, min(y_pixel, h - 1))
    width_pixel = min(width_pixel, w - x_pixel)
    height_pixel = min(height_pixel, h - y_pixel)

    if detect_white_text:
        # 提取字幕区域
        subtitle_region = frame[
            y_pixel : y_pixel + height_pixel, x_pixel : x_pixel + width_pixel
        ]

        if subtitle_region.size > 0:
            # frame是BGR格式，白色在BGR中B、G、R通道都应该是高值
            # 检测所有通道都大于阈值的像素（白色文字）
            # 使用更严格的检测：三个通道的平均值也要足够高
            bgr_avg = np.mean(subtitle_region, axis=2)
            white_mask_region = (
                (subtitle_region[:, :, 0] >= white_threshold)  # B通道
                & (subtitle_region[:, :, 1] >= white_threshold)  # G通道
                & (subtitle_region[:, :, 2] >= white_threshold)  # R通道
                & (bgr_avg >= white_threshold)  # 平均值也要高
            ).astype(np.uint8) * 255

            # 同时检测黑色边框（用于处理白色文字带黑边的情况）
            # 黑色：所有通道都小于阈值
            black_threshold = 80  # 黑色阈值
            bgr_avg_low = np.mean(subtitle_region, axis=2)
            black_mask_region = (
                (subtitle_region[:, :, 0] <= black_threshold)  # B通道
                & (subtitle_region[:, :, 1] <= black_threshold)  # G通道
                & (subtitle_region[:, :, 2] <= black_threshold)  # R通道
                & (bgr_avg_low <= black_threshold)  # 平均值也要低
            ).astype(np.uint8) * 255

            # 合并白色和黑色区域（白色文字 + 黑色边框）
            combined_mask = cv2.bitwise_or(white_mask_region, black_mask_region)

            # 使用形态学操作来连接文字像素
            # 先膨胀：扩展区域以包含边框和连接断开的文字
            # 使用更大的kernel和更多迭代次数，确保包含文字之间的空白区域
            kernel_dilate = np.ones((5, 5), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel_dilate, iterations=3)

            # MORPH_CLOSE: 先膨胀后腐蚀，连接断开的文字，填充文字之间的空隙
            kernel_close = np.ones((7, 7), np.uint8)
            combined_mask = cv2.morphologyEx(
                combined_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2
            )

            # MORPH_OPEN: 先腐蚀后膨胀，去除小的噪点
            kernel_open = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)

            # 使用高斯模糊平滑边缘，减少锯齿感
            # 先转换为float进行模糊，再转回uint8
            combined_mask_float = combined_mask.astype(np.float32) / 255.0
            combined_mask_smooth = cv2.GaussianBlur(combined_mask_float, (5, 5), 1.0)
            # 重新二值化，但保留平滑的边缘过渡
            combined_mask = (combined_mask_smooth * 255).astype(np.uint8)
            # 为了保持mask的清晰度，可以再次二值化，但使用较低的阈值以保留边缘过渡
            _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

            # 再次轻微膨胀以确保覆盖完整
            kernel_final = np.ones((3, 3), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel_final, iterations=1)

            # 将检测到的文字mask（包含白色文字、黑色边框和文字之间的区域）放回原图位置
            mask[y_pixel : y_pixel + height_pixel, x_pixel : x_pixel + width_pixel] = (
                combined_mask
            )
    else:
        # 如果不检测白色，直接使用整个矩形区域
        mask[y_pixel : y_pixel + height_pixel, x_pixel : x_pixel + width_pixel] = 255

    return mask


def detect_subtitle_region_advanced(frame, bottom_ratio=0.15):
    """
    使用更高级的方法检测字幕区域
    通过检测高对比度区域来定位字幕
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算边缘
    edges = cv2.Canny(gray, 50, 150)

    # 计算每行的边缘密度
    bottom_start = int(h * (1 - bottom_ratio))
    edge_density = np.sum(edges[bottom_start:, :], axis=1)

    # 如果底部区域有较高的边缘密度，说明可能有字幕
    if np.max(edge_density) > w * 0.1:  # 阈值可调
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[bottom_start:, :] = 255
        return mask

    # 如果没有检测到明显的字幕，返回空mask
    return np.zeros((h, w), dtype=np.uint8)


def process_frame(model, frame, mask, device, pad_out_to_modulo=8):
    """
    处理单帧图像

    Args:
        model: LaMa模型
        frame: 输入帧 (H, W, 3) BGR格式
        mask: mask (H, W) 0-255
        device: 设备
        pad_out_to_modulo: padding模数

    Returns:
        result: 处理后的帧 (H, W, 3) BGR格式
    """
    # 转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 归一化到0-1
    img = frame_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # (3, H, W)

    # 处理mask
    mask_normalized = mask.astype(np.float32) / 255.0
    mask_normalized = mask_normalized[None, ...]  # (1, H, W)

    # 记录原始尺寸
    orig_h, orig_w = img.shape[1], img.shape[2]

    # Padding到模数
    if pad_out_to_modulo > 1:
        img = pad_img_to_modulo(img, pad_out_to_modulo)
        mask_normalized = pad_img_to_modulo(mask_normalized, pad_out_to_modulo)

    # 转换为tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # (1, 3, H, W)
    mask_tensor = torch.from_numpy(mask_normalized).unsqueeze(0).float()  # (1, 1, H, W)

    # 准备batch
    batch = {"image": img_tensor, "mask": mask_tensor}

    # 推理
    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch["mask"] = (batch["mask"] > 0.5).float()
        batch = model(batch)

        # 获取结果
        result = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()

        # 裁剪回原始尺寸
        result = result[:orig_h, :orig_w]

    # 转换回BGR
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return result


def process_frames_batch(
    model, frames, masks, device, pad_out_to_modulo=8, batch_size=4
):
    """
    批量处理多帧图像（CPU优化版本）

    Args:
        model: LaMa模型
        frames: 输入帧列表 [(H, W, 3) BGR格式, ...]
        masks: mask列表 [(H, W) 0-255, ...]
        device: 设备
        pad_out_to_modulo: padding模数
        batch_size: 批处理大小（CPU建议2-4，GPU可以更大）

    Returns:
        results: 处理后的帧列表 [(H, W, 3) BGR格式, ...]
    """
    results = []

    total_batches = (len(frames) + batch_size - 1) // batch_size
    LOGGER.info(
        f"开始批量处理 {len(frames)} 帧，共 {total_batches} 个批次，每批 {batch_size} 帧"
    )

    # 按批次处理
    for batch_idx in tqdm(
        range(0, len(frames), batch_size), desc="处理批次", total=total_batches
    ):
        batch_frames = frames[batch_idx : batch_idx + batch_size]
        batch_masks = masks[batch_idx : batch_idx + batch_size]

        # 准备批次数据
        batch_images = []
        batch_mask_tensors = []
        orig_shapes = []

        for frame, mask in zip(batch_frames, batch_masks):
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 归一化到0-1
            img = frame_rgb.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # (3, H, W)

            # 处理mask
            mask_normalized = mask.astype(np.float32) / 255.0
            mask_normalized = mask_normalized[None, ...]  # (1, H, W)

            # 记录原始尺寸
            orig_h, orig_w = img.shape[1], img.shape[2]
            orig_shapes.append((orig_h, orig_w))

            # Padding到模数
            if pad_out_to_modulo > 1:
                img = pad_img_to_modulo(img, pad_out_to_modulo)
                mask_normalized = pad_img_to_modulo(mask_normalized, pad_out_to_modulo)

            batch_images.append(img)
            batch_mask_tensors.append(mask_normalized)

        # 堆叠成批次tensor
        img_tensor = torch.stack(
            [torch.from_numpy(img).float() for img in batch_images]
        )  # (B, 3, H, W)
        mask_tensor = torch.stack(
            [torch.from_numpy(mask).float() for mask in batch_mask_tensors]
        )  # (B, 1, H, W)

        # 准备batch
        batch = {"image": img_tensor, "mask": mask_tensor}

        # 批量推理
        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch["mask"] = (batch["mask"] > 0.5).float()
            batch = model(batch)

            # 处理每个结果
            batch_results = batch["inpainted"]
            for j, result in enumerate(batch_results):
                result_np = result.permute(1, 2, 0).detach().cpu().numpy()

                # 裁剪回原始尺寸
                orig_h, orig_w = orig_shapes[j]
                result_np = result_np[:orig_h, :orig_w]

                # 转换回BGR
                result_bgr = np.clip(result_np * 255, 0, 255).astype(np.uint8)
                result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_RGB2BGR)

                results.append(result_bgr)

    LOGGER.info(f"批量处理完成，共处理 {len(results)} 帧")
    return results


def extract_frames(video_path, output_dir):
    """
    从视频中提取帧

    Args:
        video_path: 视频路径
        output_dir: 输出目录

    Returns:
        frame_paths: 帧文件路径列表
        fps: 视频帧率
        width: 视频宽度
        height: 视频高度
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_paths = []
    frame_idx = 0

    LOGGER.info(f"提取视频帧: {total_frames}帧, {fps}fps, {width}x{height}")

    with tqdm(total=total_frames, desc="提取帧") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

            frame_idx += 1
            pbar.update(1)

    cap.release()

    return frame_paths, fps, width, height


def extract_audio(video_path, audio_path):
    """
    从视频中提取音频

    Args:
        video_path: 输入视频路径
        audio_path: 输出音频路径

    Returns:
        bool: 是否成功提取音频（如果视频没有音频流，返回False）
    """
    LOGGER.info(f"提取音频: {video_path} -> {audio_path}")
    try:
        # 首先检查视频是否有音频流
        check_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        check_result = subprocess.run(
            check_cmd, capture_output=True, text=True, check=False
        )

        if check_result.returncode != 0 or not check_result.stdout.strip():
            LOGGER.info("视频中没有音频流")
            return False

        # 使用ffmpeg提取音频，使用aac编码（兼容性更好）
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",  # 不包含视频
            "-acodec",
            "aac",  # 使用aac编码（兼容性更好）
            "-b:a",
            "192k",  # 音频比特率
            "-y",  # 覆盖输出文件
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        LOGGER.info(f"音频已提取到: {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        # 检查是否是"没有音频流"的错误
        error_output = e.stderr if hasattr(e, "stderr") else str(e)
        if (
            "no audio streams" in error_output.lower()
            or "stream map" in error_output.lower()
        ):
            LOGGER.info("视频中没有音频流")
            return False
        LOGGER.error(f"提取音频失败: {error_output}")
        return False
    except FileNotFoundError:
        LOGGER.error("未找到ffmpeg，请确保已安装ffmpeg")
        return False


def create_video_from_frames(
    frame_paths, output_path, fps, width, height, bitrate="8M", crf=23
):
    """
    从帧创建视频（无音频），使用ffmpeg进行高效编码

    Args:
        frame_paths: 帧文件路径列表
        output_path: 输出视频路径
        fps: 帧率
        width: 宽度
        height: 高度
        bitrate: 视频比特率（如 '8M' 表示8Mbps），默认8M
        crf: 恒定质量因子（18-28，越小质量越高，默认23）
    """
    LOGGER.info(f"创建视频: {len(frame_paths)}帧, {width}x{height}, {fps}fps")

    # 创建临时文件列表（ffmpeg需要）
    import tempfile

    temp_dir = os.path.dirname(output_path)
    temp_list_file = os.path.join(temp_dir, "frame_list.txt")

    try:
        # 写入帧文件列表
        with open(temp_list_file, "w") as f:
            for frame_path in frame_paths:
                # 使用绝对路径，确保ffmpeg能找到文件
                abs_path = os.path.abspath(frame_path)
                f.write(f"file '{abs_path}'\n")

        # 使用ffmpeg创建视频，使用h264编码（高效压缩）
        # 使用CRF模式（恒定质量）或比特率模式
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-r",
            str(fps),  # 输入帧率
            "-i",
            temp_list_file,
            "-c:v",
            "libx264",  # 使用h264编码器
            "-preset",
            "medium",  # 编码速度预设（ultrafast/fast/medium/slow）
            "-crf",
            str(crf),  # 恒定质量因子（18-28，23是默认值）
            "-pix_fmt",
            "yuv420p",  # 像素格式（兼容性最好）
            "-r",
            str(fps),  # 输出帧率
            "-y",  # 覆盖输出文件
            output_path,
        ]

        # 执行ffmpeg命令
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        LOGGER.info(f"视频已保存到: {output_path}")

    except subprocess.CalledProcessError as e:
        LOGGER.error(f"创建视频失败: {e.stderr}")
        raise
    except FileNotFoundError:
        LOGGER.error("未找到ffmpeg，请确保已安装ffmpeg")
        raise
    finally:
        # 清理临时文件
        if os.path.exists(temp_list_file):
            try:
                os.remove(temp_list_file)
            except Exception:
                pass


def merge_audio_video(video_path, audio_path, output_path, copy_video=True):
    """
    合并视频和音频

    Args:
        video_path: 视频文件路径（无音频）
        audio_path: 音频文件路径
        output_path: 最终输出视频路径
        copy_video: 是否直接复制视频流（True=快速但不压缩，False=重新编码以进一步压缩）
    """
    LOGGER.info(f"合并音频和视频: {video_path} + {audio_path} -> {output_path}")
    try:
        # 使用ffmpeg合并音频和视频
        if copy_video:
            # 直接复制视频流（快速，但文件大小不变）
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-i",
                audio_path,
                "-c:v",
                "copy",  # 复制视频流，不重新编码
                "-c:a",
                "aac",  # 使用aac编码音频（兼容性更好）
                "-b:a",
                "192k",  # 音频比特率
                "-shortest",  # 以较短的流为准
                "-y",  # 覆盖输出文件
                output_path,
            ]
        else:
            # 重新编码视频以进一步压缩（较慢，但文件更小）
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-i",
                audio_path,
                "-c:v",
                "libx264",  # 使用h264编码器
                "-preset",
                "medium",  # 编码速度预设
                "-crf",
                "23",  # 恒定质量因子
                "-c:a",
                "aac",  # 使用aac编码音频
                "-b:a",
                "192k",  # 音频比特率
                "-shortest",  # 以较短的流为准
                "-y",  # 覆盖输出文件
                output_path,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        LOGGER.info(f"最终视频已保存到: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"合并音频和视频失败: {e.stderr}")
        return False
    except FileNotFoundError:
        LOGGER.error("未找到ffmpeg，请确保已安装ffmpeg")
        return False


def load_model_config(model_dir):
    """加载模型配置，如果不存在则创建默认配置"""
    config_path = os.path.join(model_dir, "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        LOGGER.info(f"从 {config_path} 加载配置")
    else:
        # 尝试从训练配置目录加载big-lama配置
        project_root = Path(__file__).parent.parent
        training_config_path = project_root / "configs" / "training" / "big-lama.yaml"

        if training_config_path.exists():
            LOGGER.info(f"使用big-lama配置: {training_config_path}")
            # 使用Hydra的配置解析，但只加载必要的部分
            try:
                # 直接加载yaml，忽略defaults（因为那些是训练用的）
                with open(training_config_path, "r") as f:
                    config_dict = yaml.safe_load(f)
                    # 移除defaults，只保留模型相关配置
                    if "defaults" in config_dict:
                        del config_dict["defaults"]
                    train_config = OmegaConf.create(config_dict)
            except Exception as e:
                LOGGER.warning(f"加载big-lama配置失败: {e}，使用默认配置")
                train_config = create_default_config()
        else:
            LOGGER.warning(f"未找到config.yaml，使用默认配置")
            train_config = create_default_config()

    # 确保必要的配置存在
    if "training_model" not in train_config:
        train_config.training_model = OmegaConf.create({"kind": "default"})

    train_config.training_model.predict_only = True

    if "visualizer" not in train_config:
        train_config.visualizer = OmegaConf.create({"kind": "noop"})
    else:
        train_config.visualizer.kind = "noop"

    # 添加trainer配置（make_training_model需要）
    if "trainer" not in train_config:
        train_config.trainer = OmegaConf.create(
            {"kwargs": {"accelerator": None}}  # 不使用DDP
        )
    elif "kwargs" not in train_config.trainer:
        train_config.trainer.kwargs = OmegaConf.create({"accelerator": None})
    elif "accelerator" not in train_config.trainer.kwargs:
        train_config.trainer.kwargs.accelerator = None

    return train_config


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="视频去字幕脚本")
    parser.add_argument(
        "--x",
        type=float,
        default=0.00065359477124183,
        help="字幕区域左上角x坐标（归一化，0-1）",
    )
    parser.add_argument(
        "--y",
        type=float,
        default=0.7529411764705882,
        help="字幕区域左上角y坐标（归一化，0-1）",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.9952069716775599,
        help="字幕区域宽度（归一化，0-1）",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=0.05294117647058831,
        help="字幕区域高度（归一化，0-1）",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=200,
        help="白色阈值（0-255），默认200",
    )
    parser.add_argument(
        "--detect-white-text",
        action="store_true",
        default=True,
        help="是否只检测白色文字（默认True）",
    )
    parser.add_argument(
        "--no-detect-white-text",
        dest="detect_white_text",
        action="store_false",
        help="禁用白色文字检测，使用整个矩形区域",
    )
    args = parser.parse_args()

    # 获取项目根目录
    project_root = Path(__file__).parent.parent

    # 配置路径（相对于项目根目录）
    video_path = project_root / "input_videos" / "input.mp4"
    model_path = project_root / "pretrained_models" / "best.ckpt"
    output_dir = project_root / "output_videos"

    # 检查输入文件
    if not video_path.exists():
        LOGGER.error(f"输入视频不存在: {video_path}")
        return

    if not model_path.exists():
        LOGGER.error(f"模型文件不存在: {model_path}")
        return

    # 从视频文件名提取名称（不含扩展名）
    video_name = Path(video_path).stem  # 例如 "input.mp4" -> "input"

    # 创建以视频名称命名的帧文件夹
    frames_dir = project_root / "video_frames" / video_name

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 转换为字符串路径
    video_path = str(video_path)
    model_path = str(model_path)
    frames_dir = str(frames_dir)
    output_dir = str(output_dir)

    LOGGER.info(f"视频名称: {video_name}")
    LOGGER.info(f"帧保存目录: {frames_dir}")

    # 设置设备
    # 详细的GPU诊断信息
    LOGGER.info("=" * 60)
    LOGGER.info("GPU诊断信息:")
    LOGGER.info(f"PyTorch版本: {torch.__version__}")

    # 检查GPU硬件（通过lspci）
    try:
        lspci_result = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=5
        )
        if lspci_result.returncode == 0:
            nvidia_gpus = [
                line
                for line in lspci_result.stdout.split("\n")
                if "nvidia" in line.lower() or "vga" in line.lower()
            ]
            if nvidia_gpus:
                LOGGER.info(f"检测到GPU硬件: {len(nvidia_gpus)} 个设备")
                for gpu in nvidia_gpus[:3]:  # 只显示前3个
                    LOGGER.info(f"  - {gpu.strip()}")
            else:
                LOGGER.warning("未检测到NVIDIA GPU硬件（通过lspci）")
        else:
            LOGGER.warning("lspci命令执行失败")
    except FileNotFoundError:
        LOGGER.warning("lspci命令未找到")
    except Exception as e:
        LOGGER.debug(f"检查lspci时出错: {e}")

    # 检查nvidia-smi是否可用
    nvidia_smi_found = False
    try:
        # 尝试多个可能的路径
        nvidia_smi_paths = [
            "nvidia-smi",
            "/usr/bin/nvidia-smi",
            "/usr/local/bin/nvidia-smi",
        ]
        for nvidia_smi_path in nvidia_smi_paths:
            try:
                nvidia_smi_result = subprocess.run(
                    [
                        nvidia_smi_path,
                        "--query-gpu=name,memory.total",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if nvidia_smi_result.returncode == 0:
                    LOGGER.info(f"nvidia-smi输出: {nvidia_smi_result.stdout.strip()}")
                    nvidia_smi_found = True
                    break
            except FileNotFoundError:
                continue
        if not nvidia_smi_found:
            LOGGER.warning("nvidia-smi命令未找到，可能CUDA驱动未安装")
            LOGGER.warning("请检查：")
            LOGGER.warning(
                "  1. 是否安装了NVIDIA驱动: sudo apt-get install nvidia-driver-xxx"
            )
            LOGGER.warning("  2. 驱动是否正确加载: lsmod | grep nvidia")
            LOGGER.warning("  3. 是否需要重启服务器")
    except Exception as e:
        LOGGER.warning(f"检查nvidia-smi时出错: {e}")

    # 检查CUDA库路径
    cuda_paths = ["/usr/local/cuda", "/usr/local/cuda-11.8", "/usr/local/cuda-12.0"]
    cuda_found = False
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            LOGGER.info(f"找到CUDA安装路径: {cuda_path}")
            cuda_found = True
            break
    if not cuda_found:
        LOGGER.warning("未找到CUDA安装路径（/usr/local/cuda*）")

    # 检查PyTorch的CUDA支持
    cuda_available = torch.cuda.is_available()
    LOGGER.info(f"torch.cuda.is_available(): {cuda_available}")

    # 检查PyTorch编译的CUDA版本
    pytorch_cuda_version = None
    if hasattr(torch.version, "cuda") and torch.version.cuda:
        pytorch_cuda_version = torch.version.cuda
        LOGGER.info(f"PyTorch编译的CUDA版本: {pytorch_cuda_version}")

    # 检查系统CUDA版本（通过nvidia-smi）
    system_cuda_version = None
    if nvidia_smi_found:
        try:
            nvidia_smi_version = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if nvidia_smi_version.returncode == 0:
                # 尝试获取CUDA版本
                nvidia_smi_cuda = subprocess.run(
                    ["nvidia-smi"], capture_output=True, text=True, timeout=5
                )
                if nvidia_smi_cuda.returncode == 0:
                    import re

                    cuda_match = re.search(
                        r"CUDA Version: (\d+\.\d+)", nvidia_smi_cuda.stdout
                    )
                    if cuda_match:
                        system_cuda_version = cuda_match.group(1)
                        LOGGER.info(
                            f"系统CUDA版本（nvidia-smi）: {system_cuda_version}"
                        )
        except Exception:
            pass

    if cuda_available:
        try:
            if hasattr(torch.backends, "cudnn") and hasattr(
                torch.backends.cudnn, "version"
            ):
                LOGGER.info(f"cuDNN版本: {torch.backends.cudnn.version()}")
        except Exception:
            pass
    else:
        # 检查版本不匹配问题
        if pytorch_cuda_version and system_cuda_version:
            LOGGER.warning(f"⚠️  CUDA版本不匹配！")
            LOGGER.warning(f"   PyTorch编译版本: CUDA {pytorch_cuda_version}")
            LOGGER.warning(f"   系统CUDA版本: {system_cuda_version}")
            LOGGER.warning(f"   这可能导致PyTorch无法使用GPU")
        LOGGER.warning("PyTorch未检测到CUDA支持")
        LOGGER.warning("可能的原因：")
        LOGGER.warning("1. PyTorch安装的是CPU版本（需要安装CUDA版本的PyTorch）")
        LOGGER.warning("2. CUDA驱动未正确安装")
        LOGGER.warning("3. CUDA版本不匹配")
        LOGGER.warning("")
        LOGGER.warning("解决方案（按顺序执行）：")
        LOGGER.warning("")
        LOGGER.warning("步骤1: 检查GPU硬件")
        LOGGER.warning("  lspci | grep -i nvidia")
        LOGGER.warning("  或: lspci | grep -i vga")
        LOGGER.warning("")
        LOGGER.warning("步骤2: 安装NVIDIA驱动（如果nvidia-smi不可用）")
        LOGGER.warning("  # Ubuntu/Debian:")
        LOGGER.warning("  sudo apt-get update")
        LOGGER.warning("  sudo apt-get install -y nvidia-driver-535  # 或更新版本")
        LOGGER.warning("  # 安装后需要重启服务器")
        LOGGER.warning("")
        LOGGER.warning("步骤3: 验证驱动安装")
        LOGGER.warning("  nvidia-smi  # 应该显示GPU信息")
        LOGGER.warning("")
        LOGGER.warning("步骤4: 安装CUDA版本的PyTorch")
        LOGGER.warning("")
        if system_cuda_version and system_cuda_version.startswith("12."):
            LOGGER.warning("  ⚠️  检测到系统CUDA 12.x，PyTorch 1.8.0不支持CUDA 12.x")
            LOGGER.warning("")
            LOGGER.warning("  方案A（推荐）: 安装支持CUDA 12.1的PyTorch 2.x:")
            LOGGER.warning(
                "    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y"
            )
            LOGGER.warning("")
            LOGGER.warning("  方案B: 安装支持CUDA 11.8的PyTorch（向后兼容）:")
            LOGGER.warning(
                "    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"
            )
            LOGGER.warning("")
            LOGGER.warning("  方案C: 使用pip安装（CUDA 12.1）:")
            LOGGER.warning(
                "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            )
        elif system_cuda_version and system_cuda_version.startswith("11."):
            LOGGER.warning("  # 对于CUDA 11.x，安装PyTorch + CUDA 11.8:")
            LOGGER.warning(
                "  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"
            )
            LOGGER.warning("")
            LOGGER.warning("  # 或使用pip:")
            LOGGER.warning(
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )
        else:
            LOGGER.warning("  # 对于PyTorch 1.8.0 + CUDA 10.2（不推荐，版本太旧）:")
            LOGGER.warning(
                "  conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch -y"
            )
            LOGGER.warning("")
            LOGGER.warning("  # 推荐：安装更新的PyTorch + CUDA 11.8:")
            LOGGER.warning(
                "  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"
            )
            LOGGER.warning("")
            LOGGER.warning("  # 或使用pip:")
            LOGGER.warning(
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )
        LOGGER.warning("")
        LOGGER.warning("步骤5: 验证安装")
        LOGGER.warning(
            '  python -c "import torch; print(torch.cuda.is_available())"  # 应该输出True'
        )

    LOGGER.info("=" * 60)

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        LOGGER.info(f"检测到 {gpu_count} 个GPU设备")
        LOGGER.info(f"GPU 0: {gpu_name}, 显存: {gpu_memory:.2f} GB")
        device = torch.device("cuda:0")
        # GPU可以使用更大的批处理（根据显存调整）
        if gpu_memory >= 8:
            batch_size = 8  # 8GB以上显存使用8
        elif gpu_memory >= 4:
            batch_size = 4  # 4-8GB显存使用4
        else:
            batch_size = 2  # 小于4GB显存使用2
        LOGGER.info(f"GPU模式：使用批处理大小={batch_size}")
    else:
        device = torch.device("cpu")
        # 设置CPU线程数（使用所有16核）
        torch.set_num_threads(16)
        # 启用MKL优化
        if hasattr(torch.backends, "mkl"):
            torch.backends.mkl.enabled = True
        # 批处理大小（CPU建议较小，避免内存溢出）
        batch_size = 2
        LOGGER.info(f"CPU模式：使用批处理大小={batch_size}，线程数=16")

    LOGGER.info(f"最终使用设备: {device}")

    # 加载模型配置
    model_dir = os.path.dirname(model_path)
    train_config = load_model_config(model_dir)

    # 加载模型
    LOGGER.info("加载模型...")
    # 根据设备设置map_location，如果使用GPU则直接加载到GPU
    map_location = str(device) if device.type == "cuda" else "cpu"
    LOGGER.info(f"模型加载位置: {map_location}")
    # 抑制模型结构打印输出（只抑制stdout，保留stderr用于错误信息）
    with suppress_stdout():
        model = load_checkpoint(
            train_config, model_path, strict=False, map_location=map_location
        )
        model.freeze()
        if device.type == "cuda":
            model = model.to(device)
    # 确保模型在GPU上（在抑制输出之外）
    if device.type == "cuda":
        LOGGER.info(f"模型已移动到GPU: {next(model.parameters()).device}")
    model.eval()
    LOGGER.info("模型加载完成")

    # 提取音频（在处理视频之前）
    audio_path = os.path.join(output_dir, "temp_audio.aac")
    has_audio = extract_audio(video_path, audio_path)

    # 提取视频帧
    LOGGER.info("提取视频帧...")
    frame_paths, fps, width, height = extract_frames(video_path, frames_dir)

    # 处理每一帧
    LOGGER.info("处理视频帧...")
    processed_frames_dir = os.path.join(frames_dir, "processed")
    os.makedirs(processed_frames_dir, exist_ok=True)

    processed_frame_paths = []

    # 先读取所有帧和mask，分组处理
    frames_to_process = []
    masks_to_process = []
    frame_indices_to_process = []
    frames_to_skip = []  # 不需要处理的帧（mask全为0）

    LOGGER.info("读取帧和生成mask...")
    LOGGER.info(
        f"字幕区域参数: x={args.x}, y={args.y}, width={args.width}, height={args.height}"
    )
    LOGGER.info(
        f"检测参数: detect_white_text={args.detect_white_text}, white_threshold={args.white_threshold}"
    )
    for idx, frame_path in enumerate(tqdm(frame_paths, desc="准备帧")):
        frame = cv2.imread(frame_path)
        mask = detect_subtitle_region(
            frame,
            x=args.x,
            y=args.y,
            width=args.width,
            height=args.height,
            detect_white_text=args.detect_white_text,
            white_threshold=args.white_threshold,
        )

        if np.sum(mask) == 0:
            # mask全为0，跳过处理
            frames_to_skip.append((idx, frame_path, frame))
        else:
            # 需要处理的帧
            frames_to_process.append(frame)
            masks_to_process.append(mask)
            frame_indices_to_process.append((idx, frame_path))

    LOGGER.info(
        f"需要处理的帧: {len(frames_to_process)}, 跳过的帧: {len(frames_to_skip)}"
    )

    # 批量处理需要处理的帧
    if frames_to_process:
        LOGGER.info("批量处理帧...")
        processed_results = process_frames_batch(
            model,
            frames_to_process,
            masks_to_process,
            device,
            pad_out_to_modulo=8,
            batch_size=batch_size,
        )

        # 保存处理后的帧
        for (idx, frame_path), processed_frame in zip(
            frame_indices_to_process, processed_results
        ):
            frame_name = os.path.basename(frame_path)
            processed_frame_path = os.path.join(processed_frames_dir, frame_name)
            cv2.imwrite(processed_frame_path, processed_frame)
            processed_frame_paths.append((idx, processed_frame_path))

    # 保存跳过的帧（直接复制）
    for idx, frame_path, frame in frames_to_skip:
        frame_name = os.path.basename(frame_path)
        processed_frame_path = os.path.join(processed_frames_dir, frame_name)
        cv2.imwrite(processed_frame_path, frame)
        processed_frame_paths.append((idx, processed_frame_path))

    # 按原始顺序排序
    processed_frame_paths.sort(key=lambda x: x[0])
    processed_frame_paths = [path for _, path in processed_frame_paths]

    # 创建无音频的视频
    temp_video_path = os.path.join(output_dir, "temp_video_no_audio.mp4")
    LOGGER.info("合成视频（无音频）...")
    create_video_from_frames(processed_frame_paths, temp_video_path, fps, width, height)

    # 合并音频和视频
    output_video_path = os.path.join(output_dir, "output.mp4")
    if has_audio:
        LOGGER.info("合并音频和视频...")
        if merge_audio_video(temp_video_path, audio_path, output_video_path):
            # 删除临时文件
            try:
                os.remove(temp_video_path)
                os.remove(audio_path)
                LOGGER.info("已清理临时文件")
            except Exception as e:
                LOGGER.warning(f"清理临时文件失败: {e}")
        else:
            # 如果合并失败，使用无音频的视频作为输出
            LOGGER.warning("音频合并失败，使用无音频的视频")
            if os.path.exists(temp_video_path):
                os.rename(temp_video_path, output_video_path)
    else:
        # 如果没有音频，直接使用无音频的视频
        LOGGER.info("原视频没有音频，使用无音频的视频")
        if os.path.exists(temp_video_path):
            os.rename(temp_video_path, output_video_path)

    LOGGER.info("处理完成！")
    LOGGER.info(f"输出视频: {output_video_path}")


if __name__ == "__main__":
    main()

import os
import uuid
import cv2
import numpy as np
from threading import Lock
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

import onnxruntime as ort

# 导入bolus追踪模块
try:
    from bolus_tracking import BolusTracker
    BOLUS_TRACKING_AVAILABLE = True
except ImportError:
    BOLUS_TRACKING_AVAILABLE = False

# -------------------------
# Config
# -------------------------
MODEL_PATH = "mhrmseg_model.onnx"
FRAME_SIZE = (512, 512)
BASE_TEMP_DIR = "temp"  # 形如 temp/<job_id> 存放资产
os.makedirs(BASE_TEMP_DIR, exist_ok=True)

# -------------------------
# App & CORS
# -------------------------
app = FastAPI(title="VFSS Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可按需收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# ONNX Runtime Session (GPU 优先)
# -------------------------
available = ort.get_available_providers()
providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
ort_session = ort.InferenceSession(MODEL_PATH, providers=providers)

# -------------------------
# Job Manager (内存态)
# -------------------------
jobs_lock = Lock()
jobs: Dict[str, Dict[str, Any]] = {}
# 结构: jobs[job_id] = {
#   "status": "pending"|"running"|"done"|"error",
#   "error": Optional[str],
#   "frames": int,
#   "num_classes": int,
#   "summary": Dict[str, Any],
#   "dir": str
# }

############################
# Label spec per errordoc  #
############################

# 类别顺序必须与模型输出通道一致
LABELS: List[str] = [
    'UESout',     # 0 (point)
    'UESin',      # 1 (point)
    'C2',         # 2 (point)
    'C4',         # 3 (point)
    'hyoid',      # 4 (point)
    'pharynx',    # 5 (polygon)
    'vestibule',  # 6 (polygon)
    'bolus',      # 7 (polygon)
]

# 8个标签的颜色（BGR格式，按上面顺序）
# BGR格式说明：(Blue, Green, Red) - 注意顺序！
LABEL_COLORS_BGR: List[tuple] = [
    (0, 0, 255),      # 红 - UESout (B=0, G=0, R=255)
    (0, 255, 0),      # 绿 - UESin (B=0, G=255, R=0)
    (255, 0, 0),      # 蓝 - C2 (B=255, G=0, R=0)
    (0, 255, 255),    # 黄 - C4 (B=0, G=255, R=255)
    (255, 0, 255),    # 紫 - hyoid (B=255, G=0, R=255)
    (255, 255, 0),    # 青 - pharynx (B=255, G=255, R=0)
    (0, 128, 255),    # 橙 - vestibule (B=0, G=128, R=255)
    (255, 0, 128),    # 粉 - bolus (B=255, G=0, R=128)
]

POINT_CLASS_IDX: List[int] = [0, 1, 2, 3, 4]
POLY_CLASS_IDX: List[int] = [5, 6, 7]

# 推理端与评估脚本一致的归一化（0..1 尺度）
MEAN_01 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_01 = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 后处理阈值
POLY_PROB_THRESH = 0.6
POINT_PROB_THRESH = 0.6


# -------------------------
# Utils
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def crop_and_resize_frame(frame_bgr: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """
    根据帧宽/高中较小值进行中心裁剪，然后缩放到目标尺寸
    
    Args:
        frame_bgr: 输入帧 (BGR格式)
        target_size: 目标尺寸 (width, height)
    
    Returns:
        处理后的帧
    """
    h, w = frame_bgr.shape[:2]
    target_w, target_h = target_size
    
    # 选择较小的维度作为裁剪标准
    crop_size = min(w, h)
    
    # 计算裁剪区域的起始位置（中心裁剪）
    start_w = max(0, (w - crop_size) // 2)
    start_h = max(0, (h - crop_size) // 2)
    
    # 执行中心裁剪
    cropped = frame_bgr[start_h:start_h + crop_size, start_w:start_w + crop_size]
    
    # 将裁剪后的正方形图像缩放到目标尺寸
    resized = cv2.resize(cropped, target_size)
    
    return resized


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    # 与 eval_special_point.py 的 Albumentations Normalize 对齐
    # 先进行中心裁剪，再缩放到512x512
    frame_bgr = crop_and_resize_frame(frame_bgr, FRAME_SIZE)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = frame_rgb.astype(np.float32) / 255.0  # 0..1
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = (x - MEAN_01.reshape(3, 1, 1)) / STD_01.reshape(3, 1, 1)
    x = np.expand_dims(x, axis=0)   # CHW -> NCHW
    return x


def run_inference(x: np.ndarray) -> np.ndarray:
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: x})
    return outputs[0]  # (1, C, H, W)


def colorize_polygon_mask_bgr(mask_hw: np.ndarray) -> np.ndarray:
    # 仅为多边形类着色（5,6,7），其他像素保持为黑
    h, w = mask_hw.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in POLY_CLASS_IDX:
        color[mask_hw == cls_idx] = LABEL_COLORS_BGR[cls_idx]
    return color


def overlay_mask_on_frame(frame_bgr: np.ndarray,
                          polygon_mask_hw: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
    # 原VFSS为黑白，叠加到灰度底图上
    # 先进行中心裁剪，再缩放到512x512，保持与预处理一致
    frame_bgr = crop_and_resize_frame(frame_bgr, FRAME_SIZE)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color_mask_bgr = colorize_polygon_mask_bgr(polygon_mask_hw)
    overlay = cv2.addWeighted(gray_3, 1.0 - alpha, color_mask_bgr, alpha, 0.0)
    return overlay


def draw_point_markers(frame_bgr: np.ndarray,
                       per_class_logits: np.ndarray,
                       radius: int = 4,
                       thickness: int = -1,
                       prob_threshold: float = POINT_PROB_THRESH) -> np.ndarray:
    # 在叠加图上绘制点类的峰值位置
    # per_class_logits: (C, H, W)
    out = frame_bgr.copy()
    _, H, W = per_class_logits.shape
    for cls_idx in POINT_CLASS_IDX:
        logits = per_class_logits[cls_idx]
        probs = 1.0 / (1.0 + np.exp(-logits))
        total = probs.sum()
        if total <= 1e-6:
            continue
        grid_y, grid_x = np.mgrid[0:H, 0:W]
        cy = float((grid_y * probs).sum() / total)
        cx = float((grid_x * probs).sum() / total)
        max_prob = float(probs.max())
        if max_prob >= prob_threshold:
            color = LABEL_COLORS_BGR[cls_idx]
            cv2.circle(out, (int(round(cx)), int(round(cy))), radius, color, thickness)
    return out


def summarize_class_counts(class_counts_series: List[np.ndarray]) -> Dict[str, Any]:
    # class_counts_series: list of (num_classes,) 每帧每类像素数
    if not class_counts_series:
        return {"frames": 0, "num_classes": 0, "per_class_total": [], "per_class_avg": []}
    arr = np.stack(class_counts_series, axis=0)  # (T, C)
    per_class_total = arr.sum(axis=0).tolist()
    per_class_avg = arr.mean(axis=0).tolist()
    return {
        "frames": arr.shape[0],
        "num_classes": arr.shape[1],
        "per_class_total": per_class_total,
        "per_class_avg": per_class_avg,
    }

def calculate_relative_coordinates(points_data: Dict[str, Optional[Dict[str, float]]]) -> Dict[str, Any]:
    """
    计算相对坐标系中的参数
    
    相对坐标系定义：
    - 原点：C4
    - Y轴：C4→C2方向为正
    - X轴：垂直于C2C4，逆时针旋转90°为正
    
    Args:
        points_data: 包含各点绝对坐标的字典
        
    Returns:
        相对坐标系参数
    """
    # 检查必要的点是否存在
    c4_data = points_data.get('C4')
    c2_data = points_data.get('C2')
    
    if c4_data is None or c2_data is None:
        return {
            'c2c4_length': None,
            'hyoid_relative_x': None,
            'hyoid_relative_y': None,
            'hyoid_c4_distance': None,
            'ues_length': None,
            'coordinate_system_valid': False
        }
    
    # C4作为原点
    c4_x, c4_y = c4_data['x'], c4_data['y']
    c2_x, c2_y = c2_data['x'], c2_data['y']
    
    # 计算C2C4向量（Y轴方向）
    c2c4_vector_x = c2_x - c4_x
    c2c4_vector_y = c2_y - c4_y
    
    # C2C4长度（即C2的Y坐标）
    c2c4_length = np.sqrt(c2c4_vector_x**2 + c2c4_vector_y**2)
    
    # 计算单位向量
    if c2c4_length > 0:
        y_unit_x = c2c4_vector_x / c2c4_length
        y_unit_y = c2c4_vector_y / c2c4_length
        # X轴单位向量（逆时针旋转90°）
        x_unit_x = -y_unit_y
        x_unit_y = y_unit_x
    else:
        return {
            'c2c4_length': 0,
            'hyoid_relative_x': None,
            'hyoid_relative_y': None,
            'hyoid_c4_distance': None,
            'ues_length': None,
            'coordinate_system_valid': False
        }
    
    # 计算hyoid的相对坐标
    hyoid_relative_x = None
    hyoid_relative_y = None
    hyoid_c4_distance = None
    
    if points_data.get('hyoid') is not None:
        hyoid_x, hyoid_y = points_data['hyoid']['x'], points_data['hyoid']['y']
        # 计算hyoid相对于C4的向量
        hyoid_vector_x = hyoid_x - c4_x
        hyoid_vector_y = hyoid_y - c4_y
        
        # 投影到相对坐标系
        hyoid_relative_x = hyoid_vector_x * x_unit_x + hyoid_vector_y * x_unit_y
        hyoid_relative_y = hyoid_vector_x * y_unit_x + hyoid_vector_y * y_unit_y
        
        # hyoid与C4的距离
        hyoid_c4_distance = np.sqrt(hyoid_vector_x**2 + hyoid_vector_y**2)
    
    # 计算UES长度（UESout与UESin之间的距离）
    ues_length = None
    uesout_data = points_data.get('UESout')
    uesin_data = points_data.get('UESin')
    
    if uesout_data is not None and uesin_data is not None:
        uesout_x, uesout_y = uesout_data['x'], uesout_data['y']
        uesin_x, uesin_y = uesin_data['x'], uesin_data['y']
        
        ues_length = np.sqrt((uesout_x - uesin_x)**2 + (uesout_y - uesin_y)**2)
    
    return {
        'c2c4_length': float(c2c4_length),
        'hyoid_relative_x': float(hyoid_relative_x) if hyoid_relative_x is not None else None,
        'hyoid_relative_y': float(hyoid_relative_y) if hyoid_relative_y is not None else None,
        'hyoid_c4_distance': float(hyoid_c4_distance) if hyoid_c4_distance is not None else None,
        'ues_length': float(ues_length) if ues_length is not None else None,
        'coordinate_system_valid': True
    }

# -------------------------
# Background Task
# -------------------------
def process_video_job(job_id: str,
                      video_path: str,
                      save_overlays: bool,
                      poly_prob_thresh: float,
                      point_prob_thresh: float,
                      point_radius: int) -> None:
    print(f"开始处理视频任务: {job_id}")
    print(f"视频路径: {video_path}")
    print(f"保存叠加图: {save_overlays}")
    print(f"多边形阈值: {poly_prob_thresh}")
    print(f"点类阈值: {point_prob_thresh}")
    print(f"点半径: {point_radius}")
    
    with jobs_lock:
        jobs[job_id]["status"] = "running"
    job_dir = jobs[job_id]["dir"]
    frames_dir = os.path.join(job_dir, "frames")
    masks_dir = os.path.join(job_dir, "masks")
    overlays_dir = os.path.join(job_dir, "overlays")
    ensure_dir(frames_dir)
    ensure_dir(masks_dir)
    if save_overlays:
        ensure_dir(overlays_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "无法打开视频"
        return

    # 获取视频帧率，只保留整数部分
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"视频帧率: {fps} FPS")

    class_counts_series: List[np.ndarray] = []
    # 时序信号缓存
    areas_series: List[Dict[str, int]] = []
    points_series: List[Dict[str, Optional[Dict[str, float]]]] = []
    # 新增：bolus追踪相关
    bolus_masks: List[np.ndarray] = []  # 收集bolus掩膜用于追踪
    frame_idx = 0
    num_classes: Optional[int] = None

    # 规范参数范围
    try:
        poly_prob_thresh = float(np.clip(poly_prob_thresh, 0.0, 1.0))
    except Exception:
        poly_prob_thresh = POLY_PROB_THRESH
    try:
        point_prob_thresh = float(np.clip(point_prob_thresh, 0.0, 1.0))
    except Exception:
        point_prob_thresh = POINT_PROB_THRESH
    try:
        point_radius = int(max(1, int(point_radius)))
    except Exception:
        point_radius = 4

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            try:
                x = preprocess_frame(frame_bgr)
                y = run_inference(x)  # (1, C, H, W)
                _, C, H, W = y.shape
                if num_classes is None:
                    num_classes = C

                # 基于训练的多标签设置：对多边形类使用sigmoid并阈值化，而非全类argmax
                logits = y[0]  # (C, H, W)
                probs = 1.0 / (1.0 + np.exp(-logits))

                # 创建每个类的二值掩膜，允许重叠
                class_masks = {}
                for cls_idx in POLY_CLASS_IDX:
                    class_masks[cls_idx] = (probs[cls_idx] > poly_prob_thresh)

                # 创建最终的多边形掩膜（用于可视化，保持原有逻辑）
                polygon_mask_hw = np.zeros((H, W), dtype=np.uint8)
                # 使用固定顺序迭代，若像素在多个类中>阈值，则后者会覆盖前者（可按需调整优先级）
                for cls_idx in POLY_CLASS_IDX:
                    binary = class_masks[cls_idx]
                    polygon_mask_hw[binary] = cls_idx

                # 每类像素数统计（仅统计多边形类）
                counts = np.bincount(polygon_mask_hw.flatten(), minlength=len(LABELS)).astype(np.int64)
                class_counts_series.append(counts)

                # 记录多边形面积（像素）和重叠面积
                areas_series.append({
                    'pharynx': int(class_masks[5].sum()),  # 使用原始二值掩膜
                    'vestibule': int(class_masks[6].sum()),
                    'bolus': int(class_masks[7].sum()),
                    # 新增：bolus与其他区域的重叠面积（使用原始二值掩膜计算）
                    'bolus_pharynx_overlap': int((class_masks[7] & class_masks[5]).sum()),
                    'bolus_vestibule_overlap': int((class_masks[7] & class_masks[6]).sum()),
                })

                # 记录点坐标（概率质心），不足阈值则置为空
                point_entry: Dict[str, Optional[Dict[str, float]]] = {}
                for cls_idx in POINT_CLASS_IDX:
                    cls_name = LABELS[cls_idx]
                    cls_logits = logits[cls_idx]
                    cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
                    max_prob = float(cls_probs.max())
                    if max_prob < point_prob_thresh or cls_probs.sum() <= 1e-6:
                        point_entry[cls_name] = None
                    else:
                        grid_y, grid_x = np.mgrid[0:H, 0:W]
                        cy = float((grid_y * cls_probs).sum() / cls_probs.sum())
                        cx = float((grid_x * cls_probs).sum() / cls_probs.sum())
                        point_entry[cls_name] = {'x': cx, 'y': cy, 'p': max_prob}
                points_series.append(point_entry)

                # 计算相对坐标参数
                try:
                    relative_params = calculate_relative_coordinates(point_entry)
                    
                    # 将相对坐标参数添加到areas_series中
                    areas_series[-1].update({
                        'c2c4_length': relative_params['c2c4_length'],
                        'hyoid_relative_x': relative_params['hyoid_relative_x'],
                        'hyoid_relative_y': relative_params['hyoid_relative_y'],
                        'hyoid_c4_distance': relative_params['hyoid_c4_distance'],
                        'ues_length': relative_params['ues_length'],
                        'coordinate_system_valid': relative_params['coordinate_system_valid']
                    })
                except Exception as relative_error:
                    print(f"第{frame_idx}帧相对坐标计算失败: {relative_error}")
                    # 添加默认值
                    areas_series[-1].update({
                        'c2c4_length': None,
                        'hyoid_relative_x': None,
                        'hyoid_relative_y': None,
                        'hyoid_c4_distance': None,
                        'ues_length': None,
                        'coordinate_system_valid': False
                    })

                # 保存掩膜：仅包含多边形类索引
                mask_path = os.path.join(masks_dir, f"mask_{frame_idx:05d}.png")
                cv2.imwrite(mask_path, polygon_mask_hw)

                # 收集bolus掩膜用于追踪
                if BOLUS_TRACKING_AVAILABLE:
                    bolus_mask = class_masks[7].astype(np.uint8) * 255  # 转换为0-255格式
                    bolus_masks.append(bolus_mask)
                    if frame_idx % 10 == 0:  # 每10帧打印一次信息
                        bolus_area = class_masks[7].sum()
                        print(f"第{frame_idx}帧: bolus面积={bolus_area}, 掩膜已收集")
                else:
                    print("Bolus追踪模块不可用，跳过掩膜收集")

                if save_overlays:
                    # 叠加多边形着色 + 点类圆点
                    overlay = overlay_mask_on_frame(frame_bgr, polygon_mask_hw, alpha=0.5)
                    overlay = draw_point_markers(
                        overlay,
                        logits,
                        radius=point_radius,
                        thickness=-1,
                        prob_threshold=point_prob_thresh,
                    )
                    overlay_path = os.path.join(overlays_dir, f"overlay_{frame_idx:05d}.png")
                    cv2.imwrite(overlay_path, overlay)

                frame_idx += 1
                
            except Exception as frame_error:
                print(f"第{frame_idx}帧处理失败: {frame_error}")
                # 添加默认数据，确保数据完整性
                areas_series.append({
                    'pharynx': 0, 'vestibule': 0, 'bolus': 0,
                    'bolus_pharynx_overlap': 0, 'bolus_vestibule_overlap': 0,
                    'c2c4_length': None, 'hyoid_relative_x': None, 'hyoid_relative_y': None,
                    'hyoid_c4_distance': None, 'ues_length': None, 'coordinate_system_valid': False
                })
                points_series.append({})
                class_counts_series.append(np.zeros(len(LABELS), dtype=np.int64))
                
                # 保存空白掩膜
                mask_path = os.path.join(masks_dir, f"mask_{frame_idx:05d}.png")
                blank_mask = np.zeros((512, 512), dtype=np.uint8)
                cv2.imwrite(mask_path, blank_mask)
                
                if save_overlays:
                    overlay_path = os.path.join(overlays_dir, f"overlay_{frame_idx:05d}.png")
                    blank_overlay = np.zeros((512, 512, 3), dtype=np.uint8)
                    cv2.imwrite(overlay_path, blank_overlay)
                
                frame_idx += 1
                continue  # 继续处理下一帧
                
    except Exception as e:
        print(f"视频处理主循环失败: {e}")
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
        cap.release()
        return
    finally:
        cap.release()

    # 如果bolus追踪可用，则进行追踪
    bolus_flow_line = None
    if BOLUS_TRACKING_AVAILABLE and bolus_masks:
        try:
            print(f"开始bolus追踪，共{len(bolus_masks)}帧")
            # 创建bolus追踪器
            tracker = BolusTracker(min_area=100, max_distance=150.0)
            
            # 逐帧处理bolus掩膜
            for frame_idx, bolus_mask in enumerate(bolus_masks):
                try:
                    # 检测当前帧的bolus片段
                    segments = tracker.detect_bolus_segments(bolus_mask, frame_idx)
                    # 更新追踪轨迹
                    tracker.update_tracks(segments)
                except Exception as frame_error:
                    print(f"第{frame_idx}帧bolus追踪失败: {frame_error}")
                    continue  # 跳过这一帧，继续处理下一帧
            
            # 获取主流bolus的流动线
            try:
                bolus_flow_line = tracker.get_main_bolus_flow_line()
                if bolus_flow_line:
                    print(f"成功获取bolus流动线: {len(bolus_flow_line)}个点")
                else:
                    print("未检测到有效的bolus流动线")
            except Exception as flow_error:
                print(f"获取bolus流动线失败: {flow_error}")
                bolus_flow_line = None
            
        except Exception as e:
            print(f"Bolus追踪初始化失败: {e}")
            bolus_flow_line = None
    else:
        if not BOLUS_TRACKING_AVAILABLE:
            print("Bolus追踪模块不可用")
        if not bolus_masks:
            print("没有bolus掩膜数据")

    summary = summarize_class_counts(class_counts_series)
    # 将一部分可快速预览的路径（前 5 帧）
    preview = []
    limit = min(5, frame_idx)
    for i in range(limit):
        item = {"frame_index": i,
                "mask_url": f"/jobs/{job_id}/frames/{i}/mask.png"}
        if save_overlays:
            item["overlay_url"] = f"/jobs/{job_id}/frames/{i}/overlay.png"
        preview.append(item)

    with jobs_lock:
        jobs[job_id]["status"] = "done"
        jobs[job_id]["frames"] = frame_idx
        jobs[job_id]["num_classes"] = num_classes or 0
        jobs[job_id]["summary"] = {
            "frames": frame_idx,
            "num_classes": num_classes or 0,
            "per_class_total": summary.get("per_class_total", []),
            "per_class_avg": summary.get("per_class_avg", []),
            "preview": preview,
            "signals": {
                "areas": areas_series,
                "points": points_series,
            },
            # 新增：bolus流动线数据
            "bolus_flow_line": bolus_flow_line,
            # 新增：视频帧率
            "fps": fps,
        }

    # 将signals输出为CSV，便于下载
    try:
        import csv
        csv_path = os.path.join(job_dir, "signals.csv")
        fieldnames = [
            "frame",
            "area_pharynx",
            "area_vestibule",
            "area_bolus",
            "bolus_pharynx_overlap",
            "bolus_vestibule_overlap",
            # 相对坐标参数
            "c2c4_length",
            "hyoid_relative_x",
            "hyoid_relative_y", 
            "hyoid_c4_distance",
            "ues_length",
            "coordinate_system_valid",
            # bolus流动线参数
            "bolus_front_x",
            "bolus_front_y", 
            "bolus_back_x",
            "bolus_back_y",
            "bolus_track_length",
            "bolus_track_valid",
            # 绝对坐标（保留用于参考）
            "UESout_x","UESout_y","UESout_p",
            "UESin_x","UESin_y","UESin_p",
            "C2_x","C2_y","C2_p",
            "C4_x","C4_y","C4_p",
            "hyoid_x","hyoid_y","hyoid_p",
        ]
        
        print(f"开始写入CSV文件，共{frame_idx}帧")
        print(f"areas_series长度: {len(areas_series)}")
        print(f"points_series长度: {len(points_series)}")
        print(f"bolus_flow_line: {bolus_flow_line}")
        
        # 解析bolus流动线数据，为每帧分配相应的参数
        bolus_params_per_frame = {}
        if bolus_flow_line and isinstance(bolus_flow_line, dict):
            # bolus_flow_line的结构: {'front_x': [...], 'front_y': [...], 'back_x': [...], 'back_y': [...], 'frame_indices': [...]}
            front_x_list = bolus_flow_line.get('front_x', [])
            front_y_list = bolus_flow_line.get('front_y', [])
            back_x_list = bolus_flow_line.get('back_x', [])
            back_y_list = bolus_flow_line.get('back_y', [])
            frame_indices = bolus_flow_line.get('frame_indices', [])
            
            print(f"解析bolus流动线: 前端点{len(front_x_list)}个, 后端点{len(back_x_list)}个, 帧索引{len(frame_indices)}个")
            
            # 为每个有bolus追踪的帧分配参数
            for i, frame_idx_track in enumerate(frame_indices):
                if frame_idx_track < frame_idx:
                    bolus_params_per_frame[frame_idx_track] = {
                        'front_x': front_x_list[i] if i < len(front_x_list) else None,
                        'front_y': front_y_list[i] if i < len(front_y_list) else None,
                        'back_x': back_x_list[i] if i < len(back_x_list) else None,
                        'back_y': back_y_list[i] if i < len(back_y_list) else None,
                        'track_length': len(frame_indices),
                        'track_valid': True
                    }
        else:
            print(f"bolus_flow_line数据结构异常: {type(bolus_flow_line)}")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(frame_idx):
                try:
                    areas_i = areas_series[i] if i < len(areas_series) else {
                        "pharynx":0,"vestibule":0,"bolus":0,"bolus_pharynx_overlap":0,"bolus_vestibule_overlap":0,
                        "c2c4_length":None,"hyoid_relative_x":None,"hyoid_relative_y":None,
                        "hyoid_c4_distance":None,"ues_length":None,"coordinate_system_valid":False
                    }
                    points_i = points_series[i] if i < len(points_series) else {}
                    
                    # 获取当前帧的bolus流动线参数
                    bolus_params = bolus_params_per_frame.get(i, {
                        'front_x': None, 'front_y': None, 'back_x': None, 'back_y': None,
                        'track_length': None, 'track_valid': False
                    })
                    
                    row = {
                        "frame": i,
                        "area_pharynx": areas_i.get("pharynx", 0),
                        "area_vestibule": areas_series[i].get("vestibule", 0) if i < len(areas_series) else 0,
                        "area_bolus": areas_series[i].get("bolus", 0) if i < len(areas_series) else 0,
                        "bolus_pharynx_overlap": areas_series[i].get("bolus_pharynx_overlap", 0) if i < len(areas_series) else 0,
                        "bolus_vestibule_overlap": areas_series[i].get("bolus_vestibule_overlap", 0) if i < len(areas_series) else 0,
                    }
                    
                    # 添加相对坐标参数
                    row.update({
                        "c2c4_length": areas_i.get("c2c4_length", None),
                        "hyoid_relative_x": areas_i.get("hyoid_relative_x", None),
                        "hyoid_relative_y": areas_i.get("hyoid_relative_y", None),
                        "hyoid_c4_distance": areas_i.get("hyoid_c4_distance", None),
                        "ues_length": areas_i.get("ues_length", None),
                        "coordinate_system_valid": areas_i.get("coordinate_system_valid", False)
                    })
                    
                    # 添加bolus流动线参数
                    row.update({
                        "bolus_front_x": bolus_params['front_x'],
                        "bolus_front_y": bolus_params['front_y'],
                        "bolus_back_x": bolus_params['back_x'],
                        "bolus_back_y": bolus_params['back_y'],
                        "bolus_track_length": bolus_params['track_length'],
                        "bolus_track_valid": bolus_params['track_valid']
                    })
                    
                    # 添加绝对坐标
                    for name in ["UESout","UESin","C2","C4","hyoid"]:
                        val = points_i.get(name)
                        row[f"{name}_x"] = None if val is None else round(val['x'], 3)
                        row[f"{name}_y"] = None if val is None else round(val['y'], 3)
                        row[f"{name}_p"] = None if val is None else round(val['p'], 3)
                    
                    writer.writerow(row)
                except Exception as row_error:
                    print(f"写入第{i}行数据失败: {row_error}")
                    # 写入一个默认行，确保CSV格式完整
                    default_row = {field: "" for field in fieldnames}
                    default_row["frame"] = i
                    default_row["error"] = f"数据写入失败: {str(row_error)}"
                    writer.writerow(default_row)
                    continue
        
        print(f"CSV文件写入完成: {csv_path}")
        
    except Exception as csv_error:
        print(f"CSV文件写入失败: {csv_error}")
        # 即使CSV写入失败，也不影响任务完成状态

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "available_providers": ort.get_available_providers(),
        "using_providers": ort_session.get_providers(),
        "model_path": MODEL_PATH,
    }

@app.post("/analyze_video/")
async def analyze_video(background_tasks: BackgroundTasks,
                        file: UploadFile = File(...),
                        save_overlays: bool = Query(True, description="是否保存叠加可视化PNG"),
                        poly_thresh: float = Query(POLY_PROB_THRESH, ge=0.0, le=1.0, description="多边形概率阈值"),
                        point_thresh: float = Query(POINT_PROB_THRESH, ge=0.0, le=1.0, description="点类概率阈值"),
                        point_radius: int = Query(4, ge=1, le=50, description="点可视化半径")):
    # 持久化上传
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(BASE_TEMP_DIR, job_id)
    ensure_dir(job_dir)
    video_path = os.path.join(job_dir, file.filename)
    with open(video_path, "wb") as f:
        f.write(await file.read())

    with jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "error": None,
            "frames": 0,
            "num_classes": 0,
            "summary": {},
            "dir": job_dir,
        }

    background_tasks.add_task(
        process_video_job,
        job_id,
        video_path,
        save_overlays,
        float(poly_thresh),
        float(point_thresh),
        int(point_radius),
    )
    return {
        "job_id": job_id,
        "status": "pending",
        "poly_thresh": poly_thresh,
        "point_thresh": point_thresh,
        "point_radius": point_radius,
    }

@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            return JSONResponse({"error": "job_id 不存在"}, status_code=404)
        return {
            "job_id": job_id,
            "status": jobs[job_id]["status"],
            "error": jobs[job_id]["error"],
        }

@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            return JSONResponse({"error": "job_id 不存在"}, status_code=404)
        if jobs[job_id]["status"] != "done":
            return JSONResponse({"error": "任务未完成"}, status_code=409)
        return {
            "job_id": job_id,
            "frames": jobs[job_id]["frames"],
            "num_classes": jobs[job_id]["num_classes"],
            "summary": jobs[job_id]["summary"],
        }

@app.get("/jobs/{job_id}/frames/{idx}/mask.png")
def get_mask(job_id: str, idx: int):
    with jobs_lock:
        if job_id not in jobs:
            return JSONResponse({"error": "job_id 不存在"}, status_code=404)
        job_dir = jobs[job_id]["dir"]
    mask_path = os.path.join(job_dir, "masks", f"mask_{idx:05d}.png")
    if not os.path.exists(mask_path):
        return JSONResponse({"error": "帧不存在"}, status_code=404)
    return FileResponse(mask_path, media_type="image/png")


@app.get("/jobs/{job_id}/frames/{idx}/overlay.png")
def get_overlay(job_id: str, idx: int):
    with jobs_lock:
        if job_id not in jobs:
            return JSONResponse({"error": "job_id 不存在"}, status_code=404)
        job_dir = jobs[job_id]["dir"]
    overlay_path = os.path.join(job_dir, "overlays", f"overlay_{idx:05d}.png")
    if not os.path.exists(overlay_path):
        return JSONResponse({"error": "帧不存在或未保存叠加图"}, status_code=404)
    return FileResponse(overlay_path, media_type="image/png")


@app.get("/jobs/{job_id}/signals.csv")
def get_signals_csv(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            return JSONResponse({"error": "job_id 不存在"}, status_code=404)
        job_dir = jobs[job_id]["dir"]
    csv_path = os.path.join(job_dir, "signals.csv")
    if not os.path.exists(csv_path):
        return JSONResponse({"error": "signals.csv 不存在或任务未完成"}, status_code=404)
    return FileResponse(csv_path, media_type="text/csv", filename="signals.csv")

@app.get("/jobs/{job_id}/bolus_flow_line")
def get_bolus_flow_line(job_id: str):
    """获取bolus流动线数据"""
    with jobs_lock:
        if job_id not in jobs:
            return JSONResponse({"error": "job_id 不存在"}, status_code=404)
        if jobs[job_id]["status"] != "done":
            return JSONResponse({"error": "任务未完成"}, status_code=409)
        
        bolus_flow_line = jobs[job_id]["summary"].get("bolus_flow_line")
        if bolus_flow_line is None:
            return JSONResponse({"error": "bolus流动线数据不存在"}, status_code=404)
        
        return {
            "job_id": job_id,
            "bolus_flow_line": bolus_flow_line,
            "tracking_available": BOLUS_TRACKING_AVAILABLE
        }
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class BolusSegment:
    """单个bolus片段的数据结构"""
    frame_idx: int
    mask: np.ndarray  # 二值掩膜
    centroid: Tuple[float, float]  # 质心坐标
    area: int  # 面积
    bbox: Tuple[int, int, int, int]  # 边界框 (x, y, w, h)
    front_point: Optional[Tuple[float, float]] = None  # 前端点
    back_point: Optional[Tuple[float, float]] = None   # 后端点

class BolusTracker:
    """Bolus追踪器"""
    
    def __init__(self, min_area: int = 50, max_distance: float = 100.0):
        self.min_area = min_area
        self.max_distance = max_distance
        self.tracks: List[List[BolusSegment]] = []  # 多个追踪轨迹
        self.current_tracks: List[BolusSegment] = []  # 当前帧的追踪状态
    
    def detect_bolus_segments(self, bolus_mask: np.ndarray, frame_idx: int) -> List[BolusSegment]:
        """检测当前帧中的bolus片段"""
        segments = []
        
        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bolus_mask, connectivity=8)
        
        for i in range(1, num_labels):  # 跳过背景标签0
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue
                
            # 提取单个bolus的掩膜
            segment_mask = (labels == i).astype(np.uint8)
            
            # 计算质心
            y_coords, x_coords = np.where(segment_mask > 0)
            centroid = (float(np.mean(x_coords)), float(np.mean(y_coords)))
            
            # 计算边界框
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            bbox = (x, y, w, h)
            
            # 检测前端点和后端点
            front_point, back_point = self._detect_front_back_points(segment_mask, bbox)
            
            segment = BolusSegment(
                frame_idx=frame_idx,
                mask=segment_mask,
                centroid=centroid,
                area=area,
                bbox=bbox,
                front_point=front_point,
                back_point=back_point
            )
            segments.append(segment)
        
        return segments
    
    def _detect_front_back_points(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """检测bolus的前端点和后端点"""
        x, y, w, h = bbox
        
        # 假设bolus从左上往右下流动
        # 前端点：右下角（最远的右下角像素）
        # 后端点：左上角（最远的左上角像素）
        
        # 寻找右下角最远的像素
        front_candidates = []
        for i in range(max(0, y), min(mask.shape[0], y + h)):
            for j in range(max(0, x), min(mask.shape[1], x + w)):
                if mask[i, j] > 0:
                    # 计算到左上角的距离（作为前端点的评分）
                    distance = np.sqrt((i - y)**2 + (j - x)**2)
                    front_candidates.append(((j, i), distance))
        
        # 寻找左上角最远的像素
        back_candidates = []
        for i in range(max(0, y), min(mask.shape[0], y + h)):
            for j in range(max(0, x), min(mask.shape[1], x + w)):
                if mask[i, j] > 0:
                    # 计算到右下角的距离（作为后端点的评分）
                    distance = np.sqrt((i - (y + h))**2 + (j - (x + w))**2)
                    back_candidates.append(((j, i), distance))
        
        # 选择最佳的前端点和后端点
        front_point = max(front_candidates, key=lambda x: x[1])[0] if front_candidates else None
        back_point = max(back_candidates, key=lambda x: x[1])[0] if back_candidates else None
        
        return front_point, back_point
    
    def update_tracks(self, segments: List[BolusSegment]) -> None:
        """更新追踪轨迹"""
        if not self.current_tracks:
            # 第一帧，创建新的轨迹
            for segment in segments:
                self.tracks.append([segment])
            self.current_tracks = segments
            return
        
        # 为每个当前片段找到最佳匹配的轨迹
        matched_tracks = set()
        matched_segments = set()
        
        for segment in segments:
            best_track_idx = -1
            best_distance = float('inf')
            
            for track_idx, track in enumerate(self.tracks):
                if track_idx in matched_tracks:
                    continue
                
                # 计算与轨迹最后一个片段的距离
                last_segment = track[-1]
                distance = self._calculate_distance(segment, last_segment)
                
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_track_idx = track_idx
            
            if best_track_idx >= 0:
                # 添加到现有轨迹
                self.tracks[best_track_idx].append(segment)
                matched_tracks.add(best_track_idx)
                matched_segments.add(segment)
            else:
                # 创建新轨迹
                self.tracks.append([segment])
        
        # 处理未匹配的当前片段（创建新轨迹）
        for segment in segments:
            if segment not in matched_segments:
                self.tracks.append([segment])
        
        self.current_tracks = segments
    
    def _calculate_distance(self, seg1: BolusSegment, seg2: BolusSegment) -> float:
        """计算两个bolus片段之间的距离"""
        return np.sqrt((seg1.centroid[0] - seg2.centroid[0])**2 + 
                      (seg1.centroid[1] - seg2.centroid[1])**2)
    
    def identify_main_bolus_track(self) -> Optional[List[BolusSegment]]:
        """识别主流bolus轨迹"""
        if not self.tracks:
            return None
        
        # 评分标准：
        # 1. 轨迹长度（帧数）
        # 2. 轨迹连续性（平均距离变化）
        # 3. 轨迹流畅性（方向一致性）
        
        track_scores = []
        for track in self.tracks:
            if len(track) < 3:  # 至少3帧才考虑
                continue
            
            # 计算轨迹长度分数
            length_score = len(track)
            
            # 计算连续性分数（平均距离变化）
            distances = []
            for i in range(1, len(track)):
                dist = self._calculate_distance(track[i], track[i-1])
                distances.append(dist)
            continuity_score = 1.0 / (1.0 + np.std(distances)) if distances else 0
            
            # 计算流畅性分数（方向一致性）
            directions = []
            for i in range(1, len(track)):
                dx = track[i].centroid[0] - track[i-1].centroid[0]
                dy = track[i].centroid[1] - track[i-1].centroid[1]
                if dx != 0 or dy != 0:
                    angle = np.arctan2(dy, dx)
                    directions.append(angle)
            
            # 计算方向变化的标准差
            if len(directions) > 1:
                direction_changes = np.diff(directions)
                # 处理角度环绕
                direction_changes = np.abs(np.arctan2(np.sin(direction_changes), np.cos(direction_changes)))
                smoothness_score = 1.0 / (1.0 + np.std(direction_changes))
            else:
                smoothness_score = 0
            
            # 综合评分
            total_score = length_score * 0.5 + continuity_score * 0.3 + smoothness_score * 0.2
            track_scores.append((track, total_score))
        
        if not track_scores:
            return None
        
        # 返回评分最高的轨迹
        best_track = max(track_scores, key=lambda x: x[1])[0]
        return best_track
    
    def get_main_bolus_flow_line(self) -> Optional[Dict[str, List[float]]]:
        """获取主流bolus的流动线数据"""
        main_track = self.identify_main_bolus_track()
        if not main_track:
            return None
        
        # 提取前端点和后端点的坐标
        front_x = []
        front_y = []
        back_x = []
        back_y = []
        
        for segment in main_track:
            if segment.front_point:
                front_x.append(segment.front_point[0])
                front_y.append(segment.front_point[1])
            if segment.back_point:
                back_x.append(segment.back_point[0])
                back_y.append(segment.back_point[1])
        
        return {
            'front_x': front_x,
            'front_y': front_y,
            'back_x': back_x,
            'back_y': back_y,
            'frame_indices': [seg.frame_idx for seg in main_track]
        }

def analyze_bolus_flow(video_frames: List[np.ndarray], bolus_masks: List[np.ndarray]) -> Optional[Dict[str, List[float]]]:
    """分析bolus流动线的主函数"""
    tracker = BolusTracker(min_area=100, max_distance=150.0)
    
    # 逐帧处理
    for frame_idx, (frame, bolus_mask) in enumerate(zip(video_frames, bolus_masks)):
        segments = tracker.detect_bolus_segments(bolus_mask, frame_idx)
        tracker.update_tracks(segments)
    
    # 获取主流bolus的流动线
    flow_line = tracker.get_main_bolus_flow_line()
    return flow_line

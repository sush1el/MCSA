"""
Unified Marine Combat Strategy Adviser System
ESP32 Capture → OpenCV Segmentation → AI Analysis
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import requests
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk
import os
import threading
import json
import torch
import torch.nn as nn
from pathlib import Path
from strategy_nlg import StrategyNLG

# ============================================================================
# SECTION 1: OpenCV Processor (from opencv_processor.py)
# ============================================================================

from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class MapElement:
    type: str
    color: str
    position: Tuple[int, int]
    area: float
    contour: np.ndarray
    bounding_box: Tuple[int, int, int, int]

class NavalMapProcessor:
    def __init__(self):
        self.color_ranges = {
            'blue': {
                'lower': np.array([90, 80, 60]),
                'upper': np.array([130, 255, 255]),
                'name': 'water'
            },
            'brown': {
                'lower': np.array([8, 50, 50]),
                'upper': np.array([30, 255, 220]),
                'name': 'terrain'
            },
            'green': {
                'lower': np.array([35, 100, 80]),
                'upper': np.array([85, 255, 255]),
                'name': 'ally_ship'
            },
            'red': {
                'lower1': np.array([0, 120, 80]),
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 120, 80]),
                'upper2': np.array([180, 255, 255]),
                'name': 'enemy_ship'
            }
        }
        self.min_area = 800

    def process_map(self, image_path: str, visualize: bool = True):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.bilateralFilter(hsv, 9, 75, 75)

        all_masks = {}
        detected_elements = {
            'ally_ships': [],
            'enemy_ships': [],
            'terrain': [],
            'water': []
        }

        for color, params in self.color_ranges.items():
            elements, mask = self._detect_color(hsv, color, params)
            all_masks[color] = mask
            element_type = params['name']
            if element_type == 'ally_ship':
                detected_elements['ally_ships'].extend(elements)
            elif element_type == 'enemy_ship':
                detected_elements['enemy_ships'].extend(elements)
            elif element_type == 'terrain':
                detected_elements['terrain'].extend(elements)
            elif element_type == 'water':
                detected_elements['water'].extend(elements)

        segmented_img = None
        if visualize:
            segmented_img = self._create_segmented_image(img, all_masks)
            segmented_img = self._smart_fill_regions(segmented_img, all_masks)

        analysis = self._analyze_battlefield(detected_elements, img.shape)

        base_path = image_path.rsplit('.', 1)[0]
        if visualize and segmented_img is not None:
            seg_path = f"{base_path}_segmented.jpg"
            cv2.imwrite(seg_path, segmented_img)
            analysis['segmented_path'] = seg_path

        if visualize:
            vis = img.copy()
            vis = self._draw_detections(vis, detected_elements)
            vis_path = f"{base_path}_visualization.jpg"
            cv2.imwrite(vis_path, vis)
            analysis['visualization_path'] = vis_path

        return {
            'elements': detected_elements,
            'analysis': analysis,
            'image_shape': img.shape,
            'segmented_image': segmented_img
        }

    def _detect_color(self, hsv_image, color_name, params):
        elements = []

        if color_name == 'red':
            mask1 = cv2.inRange(hsv_image, params['lower1'], params['upper1'])
            mask2 = cv2.inRange(hsv_image, params['lower2'], params['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_image, params['lower'], params['upper'])

        kernel_small = np.ones((3, 3), np.uint8)
        kernel_med = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_med, iterations=1)

        kernel_dilate = np.ones((5, 5), np.uint8)
        mask_segmentation = cv2.dilate(mask, kernel_dilate, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue
            min_dim = min(w, h)
            if min_dim < 15:
                continue
            center_x = x + w // 2
            center_y = y + h // 2

            element = MapElement(
                type=params['name'],
                color=color_name,
                position=(center_x, center_y),
                area=area,
                contour=contour,
                bounding_box=(x, y, w, h)
            )
            elements.append(element)

        return elements, mask_segmentation

    def _create_segmented_image(self, original_img, masks):
        height, width = original_img.shape[:2]
        segmented = np.zeros((height, width, 3), dtype=np.uint8)
        segment_colors = {
            'blue': (255, 0, 0),
            'brown': (0, 140, 255),
            'green': (0, 255, 0),
            'red': (0, 0, 255)
        }
        color_priority = ['blue', 'brown', 'green', 'red']
        for color_name in color_priority:
            if color_name in masks and color_name in segment_colors:
                mask = masks[color_name]
                color = segment_colors[color_name]
                segmented[mask > 0] = color
        return segmented

    def _smart_fill_regions(self, segmented_img, masks):
        gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        unclassified = (gray == 0).astype(np.uint8) * 255
        if cv2.countNonZero(unclassified) == 0:
            return segmented_img
        combined_mask = np.zeros_like(unclassified)
        for mask in masks.values():
            combined_mask |= mask
        kernel = np.ones((7, 7), np.uint8)
        dist_transform = cv2.distanceTransform(unclassified, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        fill_mask = (dist_transform < 5) & (unclassified > 0)
        if np.any(fill_mask):
            kernel_fill = np.ones((5, 5), np.uint8)
            temp = segmented_img.copy()
            temp = cv2.dilate(temp, kernel_fill, iterations=2)
            segmented_img[fill_mask] = temp[fill_mask]
        return segmented_img

    def _analyze_battlefield(self, elements, image_shape):
        height, width = image_shape[:2]
        analysis = {
            'total_ally_ships': len(elements['ally_ships']),
            'total_enemy_ships': len(elements['enemy_ships']),
            'total_terrain': len(elements['terrain']),
            'ally_positions': [],
            'enemy_positions': [],
            'terrain_positions': []
        }
        for ship in elements['ally_ships']:
            analysis['ally_positions'].append({
                'x': ship.position[0],
                'y': ship.position[1],
                'area': ship.area
            })
        for ship in elements['enemy_ships']:
            analysis['enemy_positions'].append({
                'x': ship.position[0],
                'y': ship.position[1],
                'area': ship.area
            })
        for terrain in elements['terrain']:
            analysis['terrain_positions'].append({
                'x': terrain.position[0],
                'y': terrain.position[1],
                'area': terrain.area
            })
        if elements['ally_ships'] and elements['enemy_ships']:
            distances = []
            for ally in elements['ally_ships']:
                for enemy in elements['enemy_ships']:
                    dist = np.sqrt(
                        (ally.position[0] - enemy.position[0])**2 +
                        (ally.position[1] - enemy.position[1])**2
                    )
                    distances.append(dist)
            analysis['min_distance_to_enemy'] = min(distances)
            analysis['avg_distance_to_enemy'] = np.mean(distances)
        analysis['ally_formation'] = self._analyze_formation(elements['ally_ships'])
        analysis['enemy_formation'] = self._analyze_formation(elements['enemy_ships'])
        return analysis

    def _analyze_formation(self, ships):
        if len(ships) < 2:
            return "isolated" if len(ships) == 1 else "none"
        positions = np.array([ship.position for ship in ships])
        std_x = np.std(positions[:, 0])
        std_y = np.std(positions[:, 1])
        if std_x < 100 and std_y < 100:
            return "clustered"
        elif std_x > std_y * 2:
            return "horizontal_line"
        elif std_y > std_x * 2:
            return "vertical_line"
        else:
            return "scattered"

    def _draw_detections(self, img, detected_elements):
        vis = img.copy()
        for i, ship in enumerate(detected_elements['ally_ships'], 1):
            x, y, w, h = ship.bounding_box
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(vis, f"Ally {i}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        for i, ship in enumerate(detected_elements['enemy_ships'], 1):
            x, y, w, h = ship.bounding_box
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(vis, f"Enemy {i}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        for i, t in enumerate(detected_elements['terrain'], 1):
            x, y, w, h = t.bounding_box
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,140,255), 2)
            cv2.putText(vis, f"Terrain {i}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,140,255), 2)
        return vis

# ============================================================================
# SECTION 2: AI Model (from inference.py)
# ============================================================================

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class MCSAStrategyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.attention1 = SpatialAttention(32)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.attention2 = SpatialAttention(64)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.35),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared = nn.Sequential(
            nn.Linear(1024, 768),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(768),
            nn.Dropout(0.4),
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        self.primary_strategy = nn.Sequential(
            nn.Linear(256, 192),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(192),
            nn.Dropout(0.3),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 20)
        )
        
        self.engagement_distance = nn.Sequential(
            nn.Linear(256, 96),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(96),
            nn.Dropout(0.25),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )
        
        self.tempo = nn.Sequential(
            nn.Linear(256, 96),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(96),
            nn.Dropout(0.25),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        
        self.use_terrain = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )
        
        self.flanking_direction = nn.Sequential(
            nn.Linear(256, 96),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(96),
            nn.Dropout(0.25),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        
        self.force_concentration = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.retreat_threshold = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.aggression_level = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.target_priority = nn.Sequential(
            nn.Linear(256, 96),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(96),
            nn.Dropout(0.25),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        
        self.formation_type = nn.Sequential(
            nn.Linear(256, 96),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(96),
            nn.Dropout(0.25),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention1(x)
        
        x = self.conv2(x)
        x = self.attention2(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        global_features = torch.cat([avg_pool, max_pool], dim=1)
        
        shared_features = self.shared(global_features)
        
        outputs = {
            'primary_strategy': self.primary_strategy(shared_features),
            'engagement_distance': self.engagement_distance(shared_features),
            'tempo': self.tempo(shared_features),
            'use_terrain': self.use_terrain(shared_features),
            'flanking_direction': self.flanking_direction(shared_features),
            'force_concentration': self.force_concentration(shared_features),
            'retreat_threshold': self.retreat_threshold(shared_features),
            'aggression_level': self.aggression_level(shared_features),
            'target_priority': self.target_priority(shared_features),
            'formation_type': self.formation_type(shared_features)
        }
        
        return outputs

# Strategy mappings
STRATEGY_NAMES = [
    'Frontal Assault', 'Pincer Movement', 'Flanking Maneuver', 'Envelopment',
    'Feigned Retreat', 'Ambush', 'Island Hopping', 'Defensive Screen',
    'Hit and Run', 'Concentration of Force', 'Divide and Conquer', 'Crossing the T',
    'Wolf Pack', 'Decoy Operation', 'Breakthrough', 'Double Envelopment',
    'Oblique Approach', 'Echelon Formation', 'Hammer and Anvil', 'Strategic Withdrawal'
]

STRATEGY_DESCRIPTIONS = {
    'Frontal Assault': 'Direct attack with concentrated force',
    'Pincer Movement': 'Attack from two flanks simultaneously',
    'Flanking Maneuver': 'Attack enemy from the side',
    'Envelopment': 'Surround enemy forces',
    'Feigned Retreat': 'Tactical withdrawal to lure enemy',
    'Ambush': 'Concealed attack from cover',
    'Island Hopping': 'Capture strategic positions sequentially',
    'Defensive Screen': 'Protect key assets',
    'Hit and Run': 'Quick strikes and withdrawal',
    'Concentration of Force': 'Mass forces at decisive point',
    'Divide and Conquer': 'Split and defeat separately',
    'Crossing the T': 'Position broadside for maximum firepower',
    'Wolf Pack': 'Coordinated group attacks',
    'Decoy Operation': 'Mislead enemy',
    'Breakthrough': 'Penetrate enemy lines',
    'Double Envelopment': 'Encircle from both flanks',
    'Oblique Approach': 'Indirect approach',
    'Echelon Formation': 'Staggered sequential engagement',
    'Hammer and Anvil': 'Pin and attack',
    'Strategic Withdrawal': 'Organized retreat'
}

ENGAGEMENT_DISTANCE = ['Long Range', 'Medium Range', 'Close Quarters']
TEMPO = ['Aggressive', 'Measured', 'Cautious', 'Opportunistic']
FLANKING = ['No Flank', 'Left Flank', 'Right Flank', 'Both Flanks']
TARGET_PRIORITY = ['Closest', 'Strongest', 'Isolated', 'Weakest']
FORMATION = ['Wedge', 'Line', 'Circle', 'Scattered']

class ModelInference:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = MCSAStrategyModel().to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        return img_tensor
    
    def predict(self, image: np.ndarray) -> Dict:
        with torch.no_grad():
            img_tensor = self.preprocess_image(image)
            outputs = self.model(img_tensor)
            
            predictions = {
                'primary_strategy': torch.argmax(outputs['primary_strategy'], dim=1).item(),
                'engagement_distance': torch.argmax(outputs['engagement_distance'], dim=1).item(),
                'tempo': torch.argmax(outputs['tempo'], dim=1).item(),
                'use_terrain': torch.argmax(outputs['use_terrain'], dim=1).item(),
                'flanking_direction': torch.argmax(outputs['flanking_direction'], dim=1).item(),
                'force_concentration': outputs['force_concentration'].item(),
                'retreat_threshold': outputs['retreat_threshold'].item(),
                'aggression_level': outputs['aggression_level'].item(),
                'target_priority': torch.argmax(outputs['target_priority'], dim=1).item(),
                'formation_type': torch.argmax(outputs['formation_type'], dim=1).item(),
                'strategy_confidence': torch.softmax(outputs['primary_strategy'], dim=1).max().item()
            }
        
        return predictions

# ============================================================================
# SECTION 3: Unified GUI
# ============================================================================

class UnifiedMCSASystem:
    def __init__(self, root):
        self.root = root
        self.root.title("MCSA - Unified System: ESP32 → OpenCV → AI")
        self.root.geometry("1400x950")
        
        # Configuration
        self.esp32_ip = tk.StringVar(value="192.168.1.118")
        self.model_path = tk.StringVar(value="mcsa_improved_best.pth")
        self.capture_folder = "captured_maps"
        os.makedirs(self.capture_folder, exist_ok=True)
        
        # NLG Configuration
        self.nlg_mode = tk.StringVar(value="template")
        self.groq_api_key = tk.StringVar(value="")
        
        # Components
        self.processor = NavalMapProcessor()
        self.model_inference = None
        self.nlg = StrategyNLG(mode='template')
        
        # State
        self.latest_image = None
        self.segmented_image = None
        self.opencv_results = None
        
        self.setup_ui()
    
    def setup_styles(self):
        """Configure custom button and widget styles"""
        style = ttk.Style()
        
        # Primary button (Blue) - for main actions
        style.configure('Primary.TButton',
                       background='#007bff',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Arial', 10, 'bold'),
                       padding=10)
        style.map('Primary.TButton',
                 background=[('active', '#0056b3'), ('disabled', '#6c757d')])
        
        # Success button (Green) - for positive actions
        style.configure('Success.TButton',
                       background='#28a745',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Arial', 10, 'bold'),
                       padding=10)
        style.map('Success.TButton',
                 background=[('active', '#218838'), ('disabled', '#6c757d')])
        
        # Info button (Cyan) - for informational actions
        style.configure('Info.TButton',
                       background='#17a2b8',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Arial', 9),
                       padding=6)
        style.map('Info.TButton',
                 background=[('active', '#138496')])
        
        # Warning button (Orange) - for testing/validation
        style.configure('Warning.TButton',
                       background='#ffc107',
                       foreground='black',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Arial', 9),
                       padding=6)
        style.map('Warning.TButton',
                 background=[('active', '#e0a800')])
        
        # Custom LabelFrame styles
        style.configure('Header.TLabelframe', 
                       borderwidth=2, 
                       relief='groove')
        style.configure('Header.TLabelframe.Label', 
                       font=('Arial', 11, 'bold'),
                       foreground='#2c3e50')
        
        style.configure('Image.TLabelframe',
                       borderwidth=2,
                       relief='solid')
        style.configure('Image.TLabelframe.Label',
                       font=('Arial', 10, 'bold'),
                       foreground='#34495e')
        
        style.configure('Analysis.TLabelframe',
                       borderwidth=2,
                       relief='solid')
        style.configure('Analysis.TLabelframe.Label',
                       font=('Arial', 11, 'bold'),
                       foreground='#16a085')
        
        style.configure('Console.TLabelframe',
                       borderwidth=2,
                       relief='groove')
        style.configure('Console.TLabelframe.Label',
                       font=('Arial', 10, 'bold'),
                       foreground='#7f8c8d')
        
        # Radio button style
        style.configure('NLG.TRadiobutton',
                       font=('Arial', 9))
        
    def setup_ui(self):
        """Setup UI with 3-panel layout and NLG controls"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure custom styles
        self.setup_styles()
        
        # ========== CONFIGURATION SECTION ==========
        config_frame = ttk.LabelFrame(main_frame, text="⚙️ System Configuration", padding="10", style='Header.TLabelframe')
        config_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Row 0: ESP32 and Model
        ttk.Label(config_frame, text="📷 ESP32-CAM IP:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(config_frame, textvariable=self.esp32_ip, width=20, font=('Arial', 9)).grid(row=0, column=1, padx=5)
        ttk.Button(config_frame, text="Test Connection", command=self.test_esp32, style='Info.TButton').grid(row=0, column=2, padx=5)
        
        ttk.Label(config_frame, text="🤖 AI Model:", font=('Arial', 9, 'bold')).grid(row=0, column=3, sticky=tk.W, padx=(20,0))
        ttk.Entry(config_frame, textvariable=self.model_path, width=25, font=('Arial', 9)).grid(row=0, column=4, padx=5)
        ttk.Button(config_frame, text="Browse", command=self.browse_model, style='Info.TButton').grid(row=0, column=5, padx=5)
        ttk.Button(config_frame, text="⚡ Load Model", command=self.load_model, style='Success.TButton').grid(row=0, column=6, padx=5)
        
        # Row 1: Status indicators with colored backgrounds
        status_frame = ttk.Frame(config_frame)
        status_frame.grid(row=1, column=0, columnspan=7, pady=5, sticky=(tk.W, tk.E))
        
        self.esp32_status = tk.Label(status_frame, text="⚪ ESP32: Not tested", 
                                    font=('Arial', 9), bg='#e8e8e8', fg='gray',
                                    padx=10, pady=3, relief=tk.RIDGE)
        self.esp32_status.pack(side=tk.LEFT, padx=5)
        
        self.model_status = tk.Label(status_frame, text="⚪ AI Model: Not loaded", 
                                    font=('Arial', 9), bg='#e8e8e8', fg='gray',
                                    padx=10, pady=3, relief=tk.RIDGE)
        self.model_status.pack(side=tk.LEFT, padx=5)
        
        # Row 2-5: NLG Configuration
        ttk.Separator(config_frame, orient='horizontal').grid(row=2, column=0, columnspan=7, sticky=(tk.W, tk.E), pady=8)
        
        ttk.Label(config_frame, text="💬 Natural Language Generation:", font=('Arial', 9, 'bold')).grid(row=3, column=0, columnspan=2, sticky=tk.W)
        
        # Radio buttons for mode selection with custom style
        nlg_modes_frame = ttk.Frame(config_frame)
        nlg_modes_frame.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=5)
        
        ttk.Radiobutton(
            nlg_modes_frame, 
            text="📝 Templates (Offline)", 
            variable=self.nlg_mode, 
            value="template",
            command=self.change_nlg_mode,
            style='NLG.TRadiobutton'
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            nlg_modes_frame, 
            text="🦙 Ollama (Local)", 
            variable=self.nlg_mode, 
            value="ollama",
            command=self.change_nlg_mode,
            style='NLG.TRadiobutton'
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            nlg_modes_frame, 
            text="⚡ Groq (Free API)", 
            variable=self.nlg_mode, 
            value="groq",
            command=self.change_nlg_mode,
            style='NLG.TRadiobutton'
        ).pack(side=tk.LEFT, padx=5)
        
        # Groq API Key field
        api_key_frame = ttk.Frame(config_frame)
        api_key_frame.grid(row=4, column=4, columnspan=3, sticky=tk.W, padx=(10,0))
        
        ttk.Label(api_key_frame, text="🔑 Groq API Key:", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0,5))
        self.groq_key_entry = ttk.Entry(api_key_frame, textvariable=self.groq_api_key, width=25, show="*", font=('Arial', 9))
        self.groq_key_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(api_key_frame, text="Test", command=self.test_nlg_service, style='Warning.TButton').pack(side=tk.LEFT, padx=5)
        
        # NLG Status
        self.nlg_status = tk.Label(config_frame, text="✅ NLG: 📝 Template Mode (Always Available)", 
                                font=('Arial', 9, 'bold'), bg='#d4edda', fg='#155724',
                                padx=10, pady=5, relief=tk.RIDGE)
        self.nlg_status.grid(row=5, column=0, columnspan=7, pady=5, sticky=(tk.W, tk.E))
        
        # Help text
        help_frame = ttk.Frame(config_frame)
        help_frame.grid(row=6, column=0, columnspan=7, pady=3, sticky=tk.W)
        
        help_text = ttk.Label(
            help_frame,
            text="💡 Templates=Always works | Ollama=Install locally (ollama.com) | Groq=Free API (console.groq.com)",
            font=('Arial', 8, 'italic'),
            foreground="#666"
        )
        help_text.pack(side=tk.LEFT)
        
        # ========== CONTROL PANEL (CENTERED) ==========
        control_frame = ttk.LabelFrame(main_frame, text="🎮 Workflow Control", padding="5", style='Header.TLabelframe')
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Create centered button container
        button_container = ttk.Frame(control_frame)
        button_container.pack(expand=True, pady=5)

        self.capture_btn = ttk.Button(
            button_container, text="1️⃣ 📸 CAPTURE from ESP32", 
            command=self.capture_and_segment, width=28, style='Primary.TButton'
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        self.analyze_btn = ttk.Button(
            button_container, text="2️⃣ 🧠 AI ANALYZE", 
            command=self.ai_analyze, width=28, state='disabled', style='Success.TButton'
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_container, text="📁 Open Folder", 
            command=self.open_folder, width=18, style='Info.TButton'
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_container, text="📄 Load Image", 
            command=self.load_image_file, width=18, style='Info.TButton'
        ).pack(side=tk.LEFT, padx=5)
        
        # ========== IMAGE PANELS (3 PANELS) ==========
        images_frame = ttk.Frame(main_frame)
        images_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Left column: Original + Segmented
        left_column = ttk.Frame(images_frame)
        left_column.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2)
        
        # Panel 1: Original Capture
        original_frame = ttk.LabelFrame(left_column, text="1️⃣ 📷 Captured Image", padding="5", style='Image.TLabelframe')
        original_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.original_label = ttk.Label(original_frame, text="🔭 No image captured", font=('Arial', 10), foreground='gray')
        self.original_label.pack()
        
        # Panel 2: Segmented + Detection Overlay
        segmented_frame = ttk.LabelFrame(left_column, text="2️⃣ 🎯 Segmented (OpenCV) + Detection", padding="5", style='Image.TLabelframe')
        segmented_frame.pack(fill=tk.BOTH, expand=True)
        self.segmented_label = ttk.Label(segmented_frame, text="⏳ Waiting for capture", font=('Arial', 10), foreground='gray')
        self.segmented_label.pack()
        
        # Panel 3: AI Tactical Analysis (LARGE - right side)
        ai_analysis_frame = ttk.LabelFrame(images_frame, text="3️⃣ 🎯 AI Tactical Analysis", padding="10", style='Analysis.TLabelframe')
        ai_analysis_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2)
        
        # Strategy header with colored background
        header_container = tk.Frame(ai_analysis_frame, bg='#f8f9fa', relief=tk.RIDGE, bd=2)
        header_container.pack(fill=tk.X, pady=(0, 10))
        
        self.strategy_header = ttk.Label(
            ai_analysis_frame, 
            text="⏳ Waiting for analysis...",
            font=('Arial', 12, 'bold'),
            wraplength=500,
            foreground='#495057'
        )
        self.strategy_header.pack(pady=(0, 10))
                
        # Scrollable briefing text with better styling
        analysis_scroll_frame = ttk.Frame(ai_analysis_frame)
        analysis_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.strategy_report = scrolledtext.ScrolledText(
            analysis_scroll_frame,
            wrap=tk.WORD,
            font=('Consolas', 9),
            height=25,
            state='disabled',
            bg='#f8f9fa',
            fg='#212529',
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.strategy_report.pack(fill=tk.BOTH, expand=True)
        
        # ========== CONSOLE ==========
        console_frame = ttk.LabelFrame(main_frame, text="📋 System Log", padding="10", style='Console.TLabelframe')
        console_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.console = scrolledtext.ScrolledText(
            console_frame, 
            height=8, 
            state='disabled', 
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='white',
            relief=tk.FLAT,
            padx=5,
            pady=5
        )
        self.console.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(2, weight=3)
        main_frame.rowconfigure(3, weight=1)
        
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=2)
        images_frame.rowconfigure(0, weight=1)
    def log(self, message, level="INFO"):
        self.console.config(state='normal')
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.console.insert(tk.END, f"[{timestamp}] [{level}] {message}\n")
        self.console.see(tk.END)
        self.console.config(state='disabled')
    
    def update_text_widget(self, widget, text):
        widget.config(state='normal')
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, text)
        widget.config(state='disabled')
    
    def display_image(self, image_data, label_widget):
        """Display image in label widget"""
        try:
            if isinstance(image_data, str):
                img = Image.open(image_data)
            elif isinstance(image_data, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
            else:
                return
            
            max_width = 320
            max_height = 240
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo, text="")
            label_widget.image = photo
        except Exception as e:
            self.log(f"Error displaying image: {str(e)}", "ERROR")
    
    def create_segmented_with_detection_overlay(self, opencv_results):
        """Create combined image: segmented background + detection bounding boxes"""
        try:
            segmented_path = opencv_results['analysis'].get('segmented_path')
            if not segmented_path or not os.path.exists(segmented_path):
                self.log("No segmented image found for overlay", "WARNING")
                return None
            
            overlay_img = cv2.imread(segmented_path)
            elements = opencv_results['elements']
            
            # Draw ally ships (GREEN boxes)
            for i, ship in enumerate(elements['ally_ships'], 1):
                x, y, w, h = ship.bounding_box
                cv2.rectangle(overlay_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                label = f"Ally {i}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(overlay_img, (x, y-label_size[1]-8), (x+label_size[0], y), (0, 255, 0), -1)
                cv2.putText(overlay_img, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw enemy ships (RED boxes)
            for i, ship in enumerate(elements['enemy_ships'], 1):
                x, y, w, h = ship.bounding_box
                cv2.rectangle(overlay_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                label = f"Enemy {i}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(overlay_img, (x, y-label_size[1]-8), (x+label_size[0], y), (0, 0, 255), -1)
                cv2.putText(overlay_img, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw terrain (BROWN boxes)
            for i, terrain in enumerate(elements['terrain'], 1):
                x, y, w, h = terrain.bounding_box
                cv2.rectangle(overlay_img, (x, y), (x+w, y+h), (0, 140, 255), 2)
                label = f"T{i}"
                cv2.putText(overlay_img, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            
            # Save overlay - Fixed order!
            overlay_path = segmented_path.replace('_segmented.jpg', '_segmented_overlay.jpg')
            cv2.imwrite(overlay_path, overlay_img)
            
            self.log(f"✓ Created overlay: {os.path.basename(overlay_path)}")
            return overlay_path
            
        except Exception as e:
            self.log(f"Error creating overlay: {str(e)}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return None
    
    def validate_segmented_image(self, image_path: str) -> Tuple[bool, str]:
        """Validate if image contains proper segmented battlefield elements"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Cannot read image file"
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            total_pixels = img.shape[0] * img.shape[1]
            
            colors_found = {'blue': False, 'brown': False, 'green': False, 'red': False}
            color_percentages = {}
            
            # Check blue (water)
            blue_mask = cv2.inRange(hsv, np.array([90, 80, 60]), np.array([130, 255, 255]))
            blue_pixels = cv2.countNonZero(blue_mask)
            blue_percentage = (blue_pixels / total_pixels) * 100
            color_percentages['blue'] = blue_percentage
            if blue_percentage > 5:
                colors_found['blue'] = True
            
            # Check green (ally ships)
            green_mask = cv2.inRange(hsv, np.array([35, 100, 80]), np.array([85, 255, 255]))
            green_pixels = cv2.countNonZero(green_mask)
            green_percentage = (green_pixels / total_pixels) * 100
            color_percentages['green'] = green_percentage
            if green_pixels > 300 or green_percentage > 0.5:
                colors_found['green'] = True
            
            # Check red (enemy ships)
            red_mask1 = cv2.inRange(hsv, np.array([0, 120, 80]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([160, 120, 80]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_pixels = cv2.countNonZero(red_mask)
            red_percentage = (red_pixels / total_pixels) * 100
            color_percentages['red'] = red_percentage
            if red_pixels > 300 or red_percentage > 0.5:
                colors_found['red'] = True
            
            # Check if image is too dark
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 10:
                return False, "✗ Image is too dark or empty"
            
            # Validate
            errors = []
            total_colored_pixels = blue_pixels + green_pixels + red_pixels
            colored_percentage = (total_colored_pixels / total_pixels) * 100
            
            if colored_percentage < 10:
                errors.append(f"✗ Image appears mostly blank ({colored_percentage:.1f}% colored)")
            
            if not colors_found['green'] and not colors_found['red']:
                errors.append("✗ No ships detected at all")
            
            if errors:
                error_msg = "INVALID SEGMENTED IMAGE:\n\n" + "\n".join(errors)
                error_msg += f"\n\n📊 Color Coverage:"
                error_msg += f"\n  • Blue: {blue_percentage:.1f}%"
                error_msg += f"\n  • Green: {green_percentage:.2f}%"
                error_msg += f"\n  • Red: {red_percentage:.2f}%"
                return False, error_msg
            
            return True, "✓ Image validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def show_validation_error(self, error_message: str):
        """Display validation error in popup"""
        self.log(error_message, "ERROR")
        
        error_window = tk.Toplevel(self.root)
        error_window.title("⚠️ Invalid Image")
        error_window.geometry("600x400")
        error_window.transient(self.root)
        error_window.grab_set()
        
        main_frame = ttk.Frame(error_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        icon_label = ttk.Label(main_frame, text="⚠️", font=('Arial', 48))
        icon_label.pack(pady=10)
        
        title_label = ttk.Label(main_frame, text="Invalid Segmented Image", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        error_text = scrolledtext.ScrolledText(main_frame, height=12, width=70, wrap=tk.WORD, font=('Courier', 10))
        error_text.pack(pady=10, fill=tk.BOTH, expand=True)
        error_text.insert(tk.END, error_message)
        error_text.config(state='disabled')
        
        ttk.Button(main_frame, text="Close", command=error_window.destroy, style='Primary.TButton').pack(pady=10)
        
        error_window.update_idletasks()
        x = (error_window.winfo_screenwidth() // 2) - (error_window.winfo_width() // 2)
        y = (error_window.winfo_screenheight() // 2) - (error_window.winfo_height() // 2)
        error_window.geometry(f'+{x}+{y}')
    
    def test_esp32(self):
        self.log("Testing ESP32 connection...")
        
        def test():
            try:
                url = f"http://{self.esp32_ip.get()}/status"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    self.esp32_status.config(
                        text="✅ ESP32: Connected", 
                        bg='#d4edda', 
                        fg='#155724'
                    )
                    self.log("ESP32 connected successfully")
                else:
                    self.esp32_status.config(
                        text="❌ ESP32: Failed", 
                        bg='#f8d7da', 
                        fg='#721c24'
                    )
                    self.log(f"ESP32 connection failed: {response.status_code}", "ERROR")
            except Exception as e:
                self.esp32_status.config(
                    text="❌ ESP32: Failed", 
                    bg='#f8d7da', 
                    fg='#721c24'
                )
                self.log(f"ESP32 error: {str(e)}", "ERROR")
        
        threading.Thread(target=test, daemon=True).start()
    
    def browse_model(self):
        """Browse for AI model file"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=(("PyTorch Models", "*.pth"), ("All Files", "*.*"))
        )
        if filename:
            self.model_path.set(filename)
            self.log(f"Selected model: {os.path.basename(filename)}")
    
    def load_model(self):
        self.log("Loading AI model...")
        
        def load():
            try:
                model_file = self.model_path.get()
                if not os.path.exists(model_file):
                    self.log(f"Model file not found: {model_file}", "ERROR")
                    self.model_status.config(
                        text="❌ AI Model: Not found", 
                        bg='#f8d7da', 
                        fg='#721c24'
                    )
                    return
                
                self.model_inference = ModelInference(model_file)
                self.model_status.config(
                    text="✅ AI Model: Loaded", 
                    bg='#d4edda', 
                    fg='#155724'
                )
                self.log("AI model loaded successfully")
                
                total_params = sum(p.numel() for p in self.model_inference.model.parameters())
                self.log(f"Model parameters: {total_params:,}")
            except Exception as e:
                self.log(f"Error loading model: {str(e)}", "ERROR")
                self.model_status.config(
                    text="❌ AI Model: Error", 
                    bg='#f8d7da', 
                    fg='#721c24'
                )
        
        threading.Thread(target=load, daemon=True).start()
    
    def change_nlg_mode(self):
        """Handle NLG mode change"""
        mode = self.nlg_mode.get()
        
        if mode == 'template':
            self.nlg.set_mode('template')
            self.nlg_status.config(text="✅ NLG: 📝 Template Mode (Always Available)", 
                bg='#d4edda', 
                fg='#155724'
            )
            self.log("Switched to Template mode")
            
        elif mode == 'ollama':
            self.nlg.set_mode('ollama')
            if self.nlg.check_service_available():
                self.nlg_status.config(
                    text="✅ NLG: 🦙 Ollama Mode (Local LLM Running)", 
                    bg='#d4edda', 
                    fg='#155724'
                )
                self.log("Switched to Ollama mode")
            else:
                self.nlg_status.config(
                    text="⚠️ NLG: Ollama Not Running (Will Use Templates)", 
                    bg='#fff3cd', 
                    fg='#856404'
                )
                self.log("Ollama selected but not running - will fallback to templates", "WARNING")
                
        elif mode == 'groq':
            api_key = self.groq_api_key.get().strip()
            if not api_key:
                messagebox.showinfo(
                    "Groq API Key Needed",
                    "Get your FREE API key from: https://console.groq.com\n\n"
                    "Steps:\n1. Create free account\n2. Go to API Keys\n3. Create new key\n4. Paste it here\n\n"
                    "No credit card required!"
                )
                self.nlg_mode.set('template')
                return
            
            self.nlg.set_mode('groq', api_key)
            if self.nlg.check_service_available():
                self.nlg_status.config(
                    text="✅ NLG: ⚡ Groq Mode (Free API - 30 req/min)", 
                    bg='#d4edda', 
                    fg='#155724'
                )
                self.log("Switched to Groq mode")
            else:
                self.nlg_status.config(
                    text="⚠️ NLG: Groq Connection Failed (Will Use Templates)", 
                    bg='#fff3cd', 
                    fg='#856404'
                )
                self.log("Groq connection failed - will fallback to templates", "WARNING")
    
    def test_nlg_service(self):
        """Test current NLG service"""
        mode = self.nlg_mode.get()
        self.log(f"Testing {mode.upper()} service...")
        
        def test():
            try:
                result = self.nlg.test_service()
                
                if result['success']:
                    self.log(f"✓ {result['message']}")
                    messagebox.showinfo("Service Available", result['message'])
                    
                    if mode == 'template':
                        status_text = "✅ NLG: 📝 Template Mode (Always Available)"
                        bg_color = '#d4edda'
                        fg_color = '#155724'
                    elif mode == 'ollama':
                        status_text = "✅ NLG: 🦙 Ollama Mode (Local LLM Running)"
                        bg_color = '#d4edda'
                        fg_color = '#155724'
                    elif mode == 'groq':
                        status_text = "✅ NLG: ⚡ Groq Mode (Free API Connected)"
                        bg_color = '#d4edda'
                        fg_color = '#155724'
                    
                    self.nlg_status.config(text=status_text, bg=bg_color, fg=fg_color)
                    
                else:
                    self.log(f"✗ {result['message']}", "ERROR")
                    
                    error_msg = result['message']
                    if mode == 'ollama':
                        error_msg += "\n\nTo install Ollama:\n1. Visit https://ollama.com\n2. Download and install\n"
                        error_msg += "3. Run: ollama pull llama3.2\n4. Ollama runs in background automatically"
                    elif mode == 'groq':
                        error_msg += "\n\nTo get FREE Groq API key:\n1. Visit https://console.groq.com\n"
                        error_msg += "2. Sign up (no credit card)\n3. Create API key\n4. Paste key above"
                    
                    messagebox.showerror("Service Not Available", error_msg)
                    self.nlg_status.config(
                        text=f"⚠️ NLG: {mode.title()} Not Available (Using Templates)",
                        bg='#fff3cd',
                        fg='#856404'
                    )
                    
            except Exception as e:
                self.log(f"Test error: {str(e)}", "ERROR")
                messagebox.showerror("Test Failed", f"Error testing service:\n{str(e)}")
        
        threading.Thread(target=test, daemon=True).start()
    
    def _create_fallback_report(self, predictions: Dict, analysis: Dict) -> str:
        """Fallback report if NLG fails"""
        strategy_name = STRATEGY_NAMES[predictions['primary_strategy']]
        strategy_desc = STRATEGY_DESCRIPTIONS[strategy_name]
        confidence = predictions['strategy_confidence']
        
        report = f"TACTICAL STRATEGY ANALYSIS\n{'='*50}\n\n"
        report += f"PRIMARY STRATEGY:\n  {strategy_name}\n  Confidence: {confidence*100:.1f}%\n\n"
        report += f"DESCRIPTION:\n  {strategy_desc}\n\n{'='*50}\n\n"
        report += f"TACTICAL PARAMETERS:\n\n"
        report += f"Engagement:  {ENGAGEMENT_DISTANCE[predictions['engagement_distance']]}\n"
        report += f"Tempo:       {TEMPO[predictions['tempo']]}\n"
        report += f"Terrain:     {'USE cover' if predictions['use_terrain'] == 1 else 'OPEN engagement'}\n"
        report += f"Flanking:    {FLANKING[predictions['flanking_direction']]}\n\n{'='*50}\n\n"
        report += f"FORCE DEPLOYMENT:\n\n"
        report += f"Concentration: {predictions['force_concentration']*100:.0f}%\n"
        report += f"Aggression:    {predictions['aggression_level']*100:.0f}%\n"
        report += f"Retreat at:    {predictions['retreat_threshold']*100:.0f}%\n\n{'='*50}\n\n"
        report += f"RECOMMENDATIONS:\n\n"
        report += f"Target:      {TARGET_PRIORITY[predictions['target_priority']]}\n"
        report += f"Formation:   {FORMATION[predictions['formation_type']]}\n"
        
        return report
    
    def capture_and_segment(self):
        """Capture from ESP32 and create segmented overlay"""
        self.log("=" * 60)
        self.log("STEP 1: Capturing from ESP32...")
        self.capture_btn.config(state='disabled')
        
        def capture():
            try:
                url = f"http://{self.esp32_ip.get()}/capture"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'map_{timestamp}.jpg'
                    filepath = os.path.join(self.capture_folder, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    self.latest_image = filepath
                    self.log(f"✓ Image captured: {filename}")
                    self.display_image(filepath, self.original_label)
                    
                    # Segment with OpenCV
                    self.log("STEP 2: Segmenting with OpenCV...")
                    self.opencv_results = self.processor.process_map(filepath, visualize=True)
                    
                    analysis = self.opencv_results['analysis']
                    
                    # Create overlay image (segmented + detection boxes)
                    overlay_path = self.create_segmented_with_detection_overlay(self.opencv_results)
                    
                    if overlay_path and os.path.exists(overlay_path):
                        self.segmented_image = overlay_path
                        self.display_image(overlay_path, self.segmented_label)
                        self.log("✓ Segmentation + detection overlay complete")
                    elif 'segmented_path' in analysis:
                        self.segmented_image = analysis['segmented_path']
                        self.display_image(analysis['segmented_path'], self.segmented_label)
                        self.log("✓ Segmentation complete (overlay failed)")
                    
                    # Log detection summary
                    self.log("=" * 60)
                    self.log("OBJECT DETECTION RESULTS")
                    self.log(f"Allied Ships: {analysis['total_ally_ships']} (Formation: {analysis['ally_formation']})")
                    self.log(f"Enemy Ships: {analysis['total_enemy_ships']} (Formation: {analysis['enemy_formation']})")
                    self.log(f"Terrain Objects: {analysis['total_terrain']}")
                    if 'min_distance_to_enemy' in analysis:
                        self.log(f"Engagement Range - Closest: {analysis['min_distance_to_enemy']:.0f} px, Average: {analysis['avg_distance_to_enemy']:.0f} px")
                    self.log("=" * 60)
                    
                    self.log("✓ OpenCV processing complete")
                    self.log("Ready for AI analysis!")
                    
                    self.analyze_btn.config(state='normal')
                    
                else:
                    self.log(f"Capture failed: Status {response.status_code}", "ERROR")
                    
            except Exception as e:
                self.log(f"Error in capture/segment: {str(e)}", "ERROR")
                import traceback
                self.log(traceback.format_exc(), "ERROR")
            
            finally:
                self.capture_btn.config(state='normal')
        
        threading.Thread(target=capture, daemon=True).start()
    
    def ai_analyze(self):
        """AI analysis with NLG briefing generation"""
        if not self.model_inference:
            messagebox.showwarning("No Model", "Please load the AI model first!")
            self.log("AI analysis failed: Model not loaded", "ERROR")
            return
        
        if not self.segmented_image or not os.path.exists(self.segmented_image):
            messagebox.showwarning("No Image", "Please capture and segment an image first!")
            self.log("AI analysis failed: No segmented image", "ERROR")
            return
        
        self.log("=" * 60)
        self.log("Validating segmented image...")
        
        is_valid, validation_msg = self.validate_segmented_image(self.segmented_image)
        
        if not is_valid:
            self.log("Image validation FAILED", "ERROR")
            self.show_validation_error(validation_msg)
            self.log("=" * 60)
            return
        
        self.log("✓ Image validation passed")
        self.log("STEP 3: AI Strategy Analysis...")
        self.analyze_btn.config(state='disabled')
        
        def analyze():
            try:
                seg_img = cv2.imread(self.segmented_image)
                analysis = self.opencv_results['analysis']
                
                if analysis['total_ally_ships'] == 0 and analysis['total_enemy_ships'] == 0:
                    error_msg = "CANNOT ANALYZE:\n\n✗ No ships detected in the image\n\n"
                    error_msg += "The OpenCV processor found:\n"
                    error_msg += f"  • Ally ships: {analysis['total_ally_ships']}\n"
                    error_msg += f"  • Enemy ships: {analysis['total_enemy_ships']}\n"
                    error_msg += f"  • Terrain: {analysis['total_terrain']}\n\n"
                    error_msg += "Please ensure the image contains:\n"
                    error_msg += "  1. GREEN ally ship markers\n  2. RED enemy ship markers\n  3. BLUE water background"
                    
                    self.show_validation_error(error_msg)
                    self.log("Analysis aborted: No ships detected", "ERROR")
                    return
                
                # Run AI inference
                predictions = self.model_inference.predict(seg_img)
                strategy_name = STRATEGY_NAMES[predictions['primary_strategy']]
                confidence = predictions['strategy_confidence']
                
                # Generate natural language briefing
                mode = self.nlg_mode.get()
                self.log(f"Generating briefing using {mode.upper()} mode...")
                
                nlg_result = self.nlg.generate_briefing(
                    predictions=predictions,
                    opencv_analysis=analysis
                )
                
                if nlg_result['success']:
                    actual_mode = nlg_result['mode']
                    self.log(f"✓ Briefing generated using {actual_mode.upper()} mode")
                    
                    header_text = f"🎯 {strategy_name}\n📊 Confidence: {confidence*100:.0f}%"
                    self.strategy_header.config(
                        text=header_text,
                        foreground='#0c5460'
                    )
                    
                    # Create full briefing report
                    full_report = f"{'='*70}\n"
                    full_report += f"MARINE COMBAT STRATEGY ADVISER - TACTICAL BRIEFING\n"
                    full_report += f"{'='*70}\n\n"
                    full_report += f"Strategy: {strategy_name}\n"
                    full_report += f"Confidence: {confidence*100:.1f}%\n"
                    full_report += f"Generated by: {actual_mode.upper()}"
                    if actual_mode != 'template':
                        full_report += f" ({nlg_result.get('model', 'N/A')})"
                    full_report += f"\n\n{'='*70}\n\n"
                    
                    # Add natural language briefing
                    full_report += nlg_result['briefing']
                    
                    # Add tactical parameters summary
                    full_report += f"\n\n{'='*70}\n"
                    full_report += f"TACTICAL PARAMETERS SUMMARY\n"
                    full_report += f"{'='*70}\n\n"
                    full_report += f"Engagement Distance: {ENGAGEMENT_DISTANCE[predictions['engagement_distance']]}\n"
                    full_report += f"Operational Tempo:   {TEMPO[predictions['tempo']]}\n"
                    full_report += f"Terrain Usage:       {'YES - Utilize cover' if predictions['use_terrain'] == 1 else 'NO - Open engagement'}\n"
                    full_report += f"Flanking Direction:  {FLANKING[predictions['flanking_direction']]}\n"
                    full_report += f"Force Concentration: {predictions['force_concentration']*100:.0f}%\n"
                    full_report += f"Aggression Level:    {predictions['aggression_level']*100:.0f}%\n"
                    full_report += f"Retreat Threshold:   {predictions['retreat_threshold']*100:.0f}%\n"
                    full_report += f"Target Priority:     {TARGET_PRIORITY[predictions['target_priority']]}\n"
                    full_report += f"Formation:           {FORMATION[predictions['formation_type']]}\n"
                    
                    self.update_text_widget(self.strategy_report, full_report)
                    
                else:
                    self.log("NLG generation failed, using fallback", "WARNING")
                    fallback_report = self._create_fallback_report(predictions, analysis)
                    self.strategy_header.config(
                        text=f"🎖️ {strategy_name}\nConfidence: {confidence*100:.0f}%",
                        foreground='#856404'
                    )
                    self.update_text_widget(self.strategy_report, fallback_report)
                
                self.log(f"✓ AI Analysis Complete: {strategy_name} ({confidence*100:.1f}%)")
                self.log("=" * 60)
                
                # Save results
                json_path = self.segmented_image.replace('_segmented_overlay.jpg', '_ai_analysis.json')
                if json_path == self.segmented_image:
                    json_path = self.segmented_image.replace('_segmented.jpg', '_ai_analysis.json')
                
                with open(json_path, 'w') as f:
                    result_data = {
                        'timestamp': datetime.now().isoformat(),
                        'opencv_analysis': analysis,
                        'ai_strategy': {
                            'primary_strategy': strategy_name,
                            'confidence': float(confidence),
                            'engagement_distance': ENGAGEMENT_DISTANCE[predictions['engagement_distance']],
                            'tempo': TEMPO[predictions['tempo']],
                            'use_terrain': bool(predictions['use_terrain']),
                            'flanking_direction': FLANKING[predictions['flanking_direction']],
                            'force_concentration': float(predictions['force_concentration']),
                            'retreat_threshold': float(predictions['retreat_threshold']),
                            'aggression_level': float(predictions['aggression_level']),
                            'target_priority': TARGET_PRIORITY[predictions['target_priority']],
                            'formation_type': FORMATION[predictions['formation_type']]
                        },
                        'natural_language_briefing': nlg_result['briefing'] if nlg_result['success'] else None,
                        'nlg_mode': nlg_result.get('mode', 'template'),
                        'nlg_model': nlg_result.get('model', 'N/A')
                    }
                    json.dump(result_data, f, indent=2)
                
                self.log(f"✓ Results saved: {os.path.basename(json_path)}")
                
            except Exception as e:
                self.log(f"Error in AI analysis: {str(e)}", "ERROR")
                error_msg = f"AI ANALYSIS ERROR:\n\n{str(e)}\n\n"
                error_msg += "This might indicate:\n  • Corrupted model file\n"
                error_msg += "  • Incompatible image format\n  • Memory/GPU issues"
                self.show_validation_error(error_msg)
                import traceback
                self.log(traceback.format_exc(), "ERROR")
            
            finally:
                self.analyze_btn.config(state='normal')
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def load_image_file(self):
        """Load image from file for testing without ESP32"""
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=(("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*"))
        )
        if not filename:
            return
        
        self.log("=" * 60)
        self.log(f"Loading image from file: {os.path.basename(filename)}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f'loaded_{timestamp}.jpg'
        new_filepath = os.path.join(self.capture_folder, new_filename)
        
        import shutil
        shutil.copy(filename, new_filepath)
        
        self.latest_image = new_filepath
        self.display_image(new_filepath, self.original_label)
        self.log(f"✓ Image loaded")
        
        def segment():
            try:
                self.log("STEP 2: Segmenting with OpenCV...")
                self.opencv_results = self.processor.process_map(new_filepath, visualize=True)
                
                analysis = self.opencv_results['analysis']
                
                # Create overlay image
                overlay_path = self.create_segmented_with_detection_overlay(self.opencv_results)
                
                if overlay_path and os.path.exists(overlay_path):
                    self.segmented_image = overlay_path
                    self.display_image(overlay_path, self.segmented_label)
                    self.log("✓ Segmentation + detection overlay complete")
                elif 'segmented_path' in analysis:
                    self.segmented_image = analysis['segmented_path']
                    self.display_image(analysis['segmented_path'], self.segmented_label)
                    self.log("✓ Segmentation complete (overlay failed)")
                
                # Log detection summary
                self.log("=" * 60)
                self.log("OBJECT DETECTION RESULTS")
                self.log(f"Allied Ships: {analysis['total_ally_ships']} (Formation: {analysis['ally_formation']})")
                self.log(f"Enemy Ships: {analysis['total_enemy_ships']} (Formation: {analysis['enemy_formation']})")
                self.log(f"Terrain Objects: {analysis['total_terrain']}")
                self.log("=" * 60)
                
                self.log("✓ OpenCV processing complete")
                self.log("Ready for AI analysis!")
                self.analyze_btn.config(state='normal')
                
            except Exception as e:
                self.log(f"Error in segmentation: {str(e)}", "ERROR")
        
        threading.Thread(target=segment, daemon=True).start()
    
    def open_folder(self):
        """Open the captures folder"""
        import subprocess
        import platform
        
        folder_path = os.path.abspath(self.capture_folder)
        
        try:
            if platform.system() == 'Windows':
                os.startfile(folder_path)
            elif platform.system() == 'Darwin':
                subprocess.run(['open', folder_path])
            else:
                subprocess.run(['xdg-open', folder_path])
            
            self.log(f"Opened folder: {folder_path}")
        except Exception as e:
            self.log(f"Error opening folder: {str(e)}", "ERROR")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')
    
    app = UnifiedMCSASystem(root)
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║     MCSA Unified System - Ready for Exhibition              ║
╚══════════════════════════════════════════════════════════════╝

System Components:
  ✓ ESP32-CAM Integration
  ✓ OpenCV Color Segmentation  
  ✓ AI Strategy Analysis

Workflow:
  1. Configure ESP32 IP and load AI model
  2. Click "CAPTURE from ESP32" → Automatic segmentation
  3. Click "AI ANALYZE" → Get tactical recommendations

Alternative: Use "Load Image" to test without ESP32

Ready for operation!
""")
    
    root.mainloop()
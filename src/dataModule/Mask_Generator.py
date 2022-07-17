import logging

import numpy as np
import cv2
LOGGER = logging.getLogger(__name__)

class Mask_Generator:
    def __init__(self):
        self.draw_types=['line','circle','square']
        LOGGER.info('Mask_Generator Init Done')
    def irregular_mask(self,width,height):
        mask = np.zeros((height, width), np.float32)
        times=np.random.randint(0,11)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(4)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(60)
                brush_w = 5 + np.random.randint(20)
                end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
                end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
                draw_type=self.draw_types[np.random.randint(0,3)]
                if draw_type=='line':
                    cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
                elif draw_type=='circle':
                    cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
                elif draw_type=='square':
                    radius = brush_w // 2
                    mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
                start_x, start_y = end_x, end_y
        return mask

    def rectangle_mask(self,width,height):
        mask = np.zeros((height, width), np.float32)
        bbox_max_size = min(100, height - 10 * 2, width - 10 * 2)
        times = np.random.randint(0, 3)
        for i in range(times):
            box_width = np.random.randint(30, bbox_max_size)
            box_height = np.random.randint(30, bbox_max_size)
            start_x = np.random.randint(10, width - 10 - box_width + 1)
            start_y = np.random.randint(10, height - 10 - box_height + 1)
            mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
        return mask
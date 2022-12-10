#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os

import cv2
import numpy as np
import onnxruntime

from yolox.coco_classes import COCO_CLASSES
from yolox.color_list import _COLORS


class YOLOXONNX(object):
    def __init__(
        self,
        model_path='model/yolox_tiny.onnx',
        input_shape=(416, 416),
        class_score_th=0.3,
        with_p6=False,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    ):
        self.model_path=model_path
        self.input_shape=input_shape
        self.class_score_th=class_score_th
        self.with_p6=with_p6
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )
    
    def inference(self, origin_img):
        img, ratio = self.preproc(origin_img, self.input_shape)
        
        ort_inputs = {self.onnx_session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.onnx_session.run(None, ort_inputs)
        predictions = self.demo_postprocess(output[0], self.input_shape)[0]
        
        return predictions, ratio
    
    
    def preproc(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114
        
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    
    
    def demo_postprocess(self, outputs, img_size):
        grids = []
        expanded_strides = []
        
        if not self.with_p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]
            
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]
        
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        
        return outputs


class YOLOXONNX_VIS(object):
    def __init__(
        self,
        class_score_th=0.3,
    ):
        self.score_thr = class_score_th
    
    def visual(self, origin_img, predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = self.vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=self.score_thr, class_names=COCO_CLASSES)
        
        return origin_img
    
    def multiclass_nms(self, boxes, scores, nms_thr, score_thr, class_agnostic=True):
        """Multiclass NMS implemented in Numpy"""
        if class_agnostic:
            nms_method = self.multiclass_nms_class_agnostic
        else:
            nms_method = self.multiclass_nms_class_aware
        return nms_method(boxes, scores, nms_thr, score_thr)
    
    
    def multiclass_nms_class_aware(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-aware version."""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)
    
    
    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
        
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets
    
    
    def vis(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            
            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            
            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        
        return img
    
    
    def nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]
        
        return keep
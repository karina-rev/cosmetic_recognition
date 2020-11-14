import config
import os
import cv2
import numpy as np
from openvino.inference_engine import IECore

SOS_INDEX = 0
EOS_INDEX = 1
MAX_SEQ_LEN = 28
PROB_THRESHOLD = 0.5
ALPHABET = "  0123456789abcdefghijklmnopqrstuvwxyz"
DEVICE = 'CPU'


class TextSpotting():
    """ Text Spotting Class """

    def __init__(self):
        self.mask_rcnn_exec_net = None
        self.text_enc_exec_net = None
        self.text_dec_exec_net = None
        self.hidden_shape = None
        self.n = None
        self.c = None
        self.h = None
        self.w = None

    def train_network(self):
        ie = IECore()
        mask_rcnn_net = ie.read_network(model=config.PATH_TO_MASK_RCNN_MODEL,
                                        weights=os.path.splitext(config.PATH_TO_MASK_RCNN_MODEL)[0] + '.bin')
        text_enc_net = ie.read_network(model=config.PATH_TO_TEXT_ENC_MODEL,
                                       weights=os.path.splitext(config.PATH_TO_TEXT_ENC_MODEL)[0] + '.bin')
        text_dec_net = ie.read_network(model=config.PATH_TO_TEXT_DEC_MODEL,
                                       weights=os.path.splitext(config.PATH_TO_TEXT_DEC_MODEL)[0] + '.bin')

        self.mask_rcnn_exec_net = ie.load_network(network=mask_rcnn_net, device_name=DEVICE, num_requests=2)
        self.text_enc_exec_net = ie.load_network(network=text_enc_net, device_name=DEVICE)
        self.text_dec_exec_net = ie.load_network(network=text_dec_net, device_name=DEVICE)

        self.hidden_shape = text_dec_net.input_info['prev_hidden'].input_data.shape
        self.n, self.c, self.h, self.w = mask_rcnn_net.input_info['im_data'].input_data.shape

        del mask_rcnn_net
        del text_enc_net
        del text_dec_net
        return self

    def search_text(self, image):
        if None in (self.mask_rcnn_exec_net, self.text_dec_exec_net, self.text_enc_exec_net,
                    self.hidden_shape, self.n, self.c, self.h, self.w):
            self.train_network()

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        scale_x = self.w / image.shape[1]
        scale_y = self.h / image.shape[0]
        input_image = cv2.resize(image, (self.w, self.h))

        input_image_size = input_image.shape[:2]
        input_image = np.pad(input_image, ((0, self.h - input_image_size[0]),
                                           (0, self.w - input_image_size[1]),
                                           (0, 0)),
                             mode='constant', constant_values=0)
        # Change data layout from HWC to CHW.
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((self.n, self.c, self.h, self.w)).astype(np.float32)
        input_image_info = np.asarray([[input_image_size[0], input_image_size[1], 1]], dtype=np.float32)

        # Run the net.
        outputs = self.mask_rcnn_exec_net.infer({'im_data': input_image, 'im_info': input_image_info})

        # Parse detection results of the current request
        boxes = outputs['boxes']
        scores = outputs['scores']
        classes = outputs['classes'].astype(np.uint32)
        raw_masks = outputs['raw_masks']
        text_features = outputs['text_features']

        # Filter out detections with low confidence.
        detections_filter = scores > PROB_THRESHOLD
        classes = classes[detections_filter]
        boxes = boxes[detections_filter]
        raw_masks = raw_masks[detections_filter]
        text_features = text_features[detections_filter]

        boxes[:, 0::2] /= scale_x
        boxes[:, 1::2] /= scale_y
        masks = []
        for box, cls, raw_mask in zip(boxes, classes, raw_masks):
            raw_cls_mask = raw_mask[cls, ...]
            mask = segm_postprocess(box, raw_cls_mask, image.shape[0], image.shape[1])
            masks.append(mask)

        texts = []
        for feature in text_features:
            feature = self.text_enc_exec_net.infer({'input': feature})['output']
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))

            hidden = np.zeros(self.hidden_shape)
            prev_symbol_index = np.ones((1,)) * SOS_INDEX

            text = ''
            for i in range(MAX_SEQ_LEN):
                decoder_output = self.text_dec_exec_net.infer({
                    'prev_symbol': prev_symbol_index,
                    'prev_hidden': hidden,
                    'encoder_outputs': feature})
                symbols_distr = decoder_output['output']
                prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
                if prev_symbol_index == EOS_INDEX:
                    break
                text += ALPHABET[prev_symbol_index]
                hidden = decoder_output['hidden']

            texts.append(text)

        return texts


def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp


def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    # Put an object mask in an image mask.
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                            (x0 - extended_box[0]):(x1 - extended_box[0])]
    return im_mask
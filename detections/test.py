import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from utils.utils import cvtColor, get_classes, get_new_img_size, resize_image, preprocess_input
from torchvision.ops import nms
import colorsys


def prepare_img(img_path):
    image = Image.open(img_path)
    image_shape = np.array(np.shape(image)[0:2])
    input_shape = get_new_img_size(image_shape[0], image_shape[1])
    image = cvtColor(image)
    image_data = resize_image(image, [input_shape[1], input_shape[0]])
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    return image_data, image, image_shape, input_shape


def post_process(boxes, labels, scores, image_shape, input_shape, nms_iou, confidence):
    boxes[..., [0, 2]] = (boxes[..., [0, 2]]) / input_shape[1]
    boxes[..., [1, 3]] = (boxes[..., [1, 3]]) / input_shape[0]

    selected = scores > confidence
    selected_boxes = boxes[selected]
    selected_labels = labels[selected]
    selected_scores = scores[selected]

    final_boxes = []
    final_labels = []
    unique_cls = torch.unique(selected_labels)
    for l in unique_cls:
        cls_mask = selected_labels == l

        cls_boxes = selected_boxes[cls_mask]
        cls_scores = selected_scores[cls_mask]
        if cls_boxes.shape[0] > 0:
            keep = nms(cls_boxes, cls_scores, nms_iou)

        keeped_boxes = cls_boxes[keep]
        keeped_scores = cls_scores[keep]
        keeped_labels = l * torch.ones((len(keep)), dtype=int)

        final_boxes.append(torch.cat([keeped_boxes, keeped_scores.unsqueeze(-1)], dim=-1))
        final_labels.append(keeped_labels)

    final_boxes = torch.cat(final_boxes, dim=0)
    final_labels = torch.cat(final_labels, dim=0)

    if final_boxes.shape[0] > 0:
        final_boxes[..., [0, 2]] = final_boxes[..., [0, 2]] * image_shape[1]
        final_boxes[..., [1, 3]] = final_boxes[..., [1, 3]] * image_shape[0]

    return final_boxes, final_labels


if __name__ == '__main__':
    img_path = r'C:\Users\nick1\Desktop\files\codehub\mid-term\muse\imgs\ph.jpg'
    inputs, image, image_shape, input_shape = prepare_img(img_path)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
    model.eval()

    class_names, num_classes = get_classes(classes_path=r'C:\Users\nick1\Desktop\files\codehub\mid-term\muse\det_classes.txt')

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    font = ImageFont.truetype(font=r'C:\Users\nick1\Desktop\files\codehub\mid-term\muse\simhei.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))

    with torch.no_grad():
        inputs = torch.from_numpy(inputs)
        outputs = model(inputs)
        output = outputs[0]
        boxes = output['boxes']
        labels = output['labels']
        scores = output['scores']
        final_boxes, final_labels = post_process(boxes, labels, scores, image_shape, input_shape, nms_iou=0.3, confidence=0.5)

        for i, l in list(enumerate(final_labels)):
            label = class_names[int(l)-2]
            if label == 'toothbrush':
                label = 'person'

            box_score = final_boxes[i]
            left, top, right, bottom, score = box_score.numpy()
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(label, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for t in range(thickness):
                draw.rectangle([left + t, top + t, right - t, bottom - t], outline=colors[l])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[l])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

    image.save('frcnn.jpg')

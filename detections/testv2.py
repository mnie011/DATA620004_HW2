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
    img_path = r'C:\Users\nick1\Desktop\files\codehub\mid-term\muse\imgs\road.jpg'
    inputs, image, image_shape, input_shape = prepare_img(img_path)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
    model.eval()
    transform = model.transform
    backbone = model.backbone
    rpn = model.rpn

    thickness = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))

    with torch.no_grad():
        inputs = torch.from_numpy(inputs)

        inputs, _ = transform(inputs)
        features = backbone(inputs.tensors)
        proposals, _ = rpn(inputs, features)
        proposals = proposals[0]

        proposals[..., [0, 2]] = proposals[..., [0, 2]] / input_shape[1] * image_shape[1]
        proposals[..., [1, 3]] = proposals[..., [1, 3]] / input_shape[0] * image_shape[0]
        # proposals = proposals[:500]
        for proposal in proposals:
            left, top, right, bottom = proposal.numpy()
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            draw = ImageDraw.Draw(image)

            for t in range(thickness):
                draw.rectangle([left + t, top + t, right - t, bottom - t], outline=(255, 0, 0))
            del draw
    image.save('bike.jpg')

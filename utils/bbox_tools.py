import numpy as np 



def enlarge_bboxes(bbox, shape):
    # bbox shape: [x1, y1, x2, y2, score]
    # assert isinstance(bbox, np.ndarray), '{} is not supported.'.format(type(bbox))
    x1, y1, x2, y2 = bbox[:4]
    cen_x = (x1 + x2) / 2
    cen_y = (y1 + y2) / 2
    w = (x2 - x1 + 1) / 2
    h = (y2 - y1 + 1) / 2

    x1 = max(0, int(cen_x - w * 1.2))
    y1 = max(0, int(cen_y - h * 1.2))
    x2 = min(shape[1], cen_x + w * 1.25)
    y2 = min(shape[0], cen_y + h * 1.15)
    
    return [x1, y1, x2, y2, bbox[-2], bbox[-1]]


def face_track(pre_bboxes, pre_result, bboxes, iou_thres=0.5):
    assert len(pre_bboxes) > 0
    num_pre = len(pre_bboxes)
    _pre_bboxes = np.array(pre_bboxes).reshape(num_pre, 6)
    x1 = _pre_bboxes[:, 0]
    y1 = _pre_bboxes[:, 1]
    x2 = _pre_bboxes[:, 2]
    y2 = _pre_bboxes[:, 3]
    widths = x2 - x1
    heights = y2 - y1
    area = heights * widths

    pre_frame_bboxes = []
    pre_frame_result = []
    unknown_bboxes = []
    for bbox in bboxes:
        xx1 = np.maximum(bbox[0], x1)
        yy1 = np.maximum(bbox[1], y1)
        xx2 = np.minimum(bbox[2], x2)
        yy2 = np.minimum(bbox[3], y2)
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area
        idx = np.argmax(overlap)
        if overlap[idx] > iou_thres and pre_result[idx] != 'UNKNOWN':
            bbox[-2] = _pre_bboxes[idx, -2]
            pre_frame_bboxes.append(bbox)
            pre_frame_result.append(pre_result[idx])
        else:
            unknown_bboxes.append(bbox)
    return pre_frame_bboxes, pre_frame_result, unknown_bboxes



def NMS(boxes, overlap_threshold):
    '''

    :param boxes: numpy nx5, n is the number of boxes, 0:4->x1, y1, x2, y2, 4->score
    :param overlap_threshold:
    :return:
    '''
    if boxes.shape[0] == 0:
        return boxes

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype != np.float32:
        boxes = boxes.astype(np.float32)

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    sc = boxes[:, 4]
    widths = x2 - x1
    heights = y2 - y1

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = heights * widths
    idxs = np.argsort(sc)  # 从小到大排序

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # compare secend highest score boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bo（ box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]
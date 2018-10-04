import random
import numpy as np
import cv2

__all__ = ['to_tuple', 'BasicTransform', 'DualTransform', 'ImageOnlyTransform']


def to_tuple(param, low=None):
    if isinstance(param, tuple):
        return param
    else:
        return (-param if low is None else low, param)


class BasicTransform(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, **kwargs):
        if random.random() < self.p:
            params = self.get_params()
            params = self.update_params(params, **kwargs)
            res = {}
            for key, arg in kwargs.items():
                target_function = self.targets.get(key, lambda x, **p: x)
                target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            return res
        return kwargs

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

    def update_params(self, params, **kwargs):
        if hasattr(self, 'interpolation'):
            params['interpolation'] = self.interpolation
        params.update({'cols': kwargs['image'].shape[1], 'rows': kwargs['image'].shape[0]})
        return params

    @property
    def target_dependence(self):
        return {'bboxes': ['image',]}


class DualTransform(BasicTransform):
    """Transform for segmentation task."""

    @property
    def targets(self):
        return {'image': self.apply, 'mask': self.apply_to_mask, 'bboxes': self.apply_to_bboxes}

    def apply_to_bboxes_generic(self, bboxes, image, **params):
        new_bboxes = []

        # TODO we can reduce the number of times self.apply is ran by drawing multiple bboxs per mask
        # if len(bboxes) > 255:
        #     raise NotImplementedError

        for i, bbox in enumerate(bboxes):
            mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

            # generate a mask using the bbox at the same size as the image
            p1 = (int(image.shape[1] * bbox[0]), int(image.shape[0] * bbox[1]))
            p2 = (int(image.shape[1] * bbox[2]), int(image.shape[0] * bbox[3]))
            cv2.rectangle(mask, p1, p2, (255), -1)

            # apply the operation
            mask = self.apply(mask, **{k: cv2.INTER_NEAREST if k == 'interpolation' else v for k, v in params.items()})

        # for i, bbox in enumerate(bboxes):
        #     thresh = cv2.inRange(mask, i+1, i+1)
            # detect the box (or now maybe multiple) in the new image
            _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # add new boxes to return list
            applied_bboxes = []
            for c in contours:
                rect = cv2.boundingRect(c)
                rect = [rect[0] / mask.shape[1],
                        rect[1] / mask.shape[0],
                        (rect[0] + rect[2]) / mask.shape[1],
                        (rect[1] + rect[3]) / mask.shape[0],
                        bbox[4]
                        ]
                applied_bboxes.append(rect)

            new_bboxes.extend(applied_bboxes)
        return new_bboxes

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def apply_to_bboxes(self, bboxes, **params):
        bboxes = [list(bbox) for bbox in bboxes]
        image = params.pop('image', None)
        try:
            return [self.apply_to_bbox(bbox[:4], **params) + bbox[4:] for bbox in bboxes]
        except NotImplementedError as e:
            if image is not None:
                return self.apply_to_bboxes_generic(bboxes, image, **params)
            else:
                raise e


    def apply_to_mask(self, img, **params):
        return self.apply(img, **{k: cv2.INTER_NEAREST if k == 'interpolation' else v for k, v in params.items()})


class ImageOnlyTransform(BasicTransform):
    """Transform applied to image only."""

    @property
    def targets(self):
        return {'image': self.apply}


class NoOp(DualTransform):
    """Does nothing"""

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, img, **params):
        return img

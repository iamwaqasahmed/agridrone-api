# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model="yolo11n-seg.pt", source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        # tuple if PyTorch model or array if exported
        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1] # preds[1][-1].shape :-> torch.Size([1, 32, 120, 160])
        return super().postprocess(preds[0], img, orig_imgs, protos=protos) # preds[0].shape :-> torch.Size([1, 43, 6300])

    def construct_results(self, preds, img, orig_imgs, protos):
        """
        Constructs a list of result objects from the predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.
            protos (List[torch.Tensor]): List of prototype masks.

        Returns:
            (list): List of result objects containing the original images, image paths, class names, bounding boxes, and masks.
        """
        print(f'***** Calling def construct_results() in models/yolo/segment.py *****')
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """
        Constructs the result object from the prediction.

        Args:
            pred (np.ndarray): The predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.

        Returns:
            (Results): The result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        # assert 0, 'in def construct_result ' 
        print(f'len(pred): {len(pred)} in def construct_result')
        if not len(pred):  # save empty boxes
            masks = None
            print(f'Len of masks == 0 in def construct_results in yolo segment predict.py')
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 7:], pred[:, :4], orig_img.shape[:2])  # HWC # '6' changed to '7' since masks start at one index later than previously
        else:
            masks = ops.process_mask(proto, pred[:, 7:], pred[:, :4], img.shape[2:], upsample=True)  # HWC # HWC # '6' changed to '7' since masks start at one index later than previously
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :7], masks=masks) # '6' changed to '7' since a box now contains
        # 4*coords, growth_val, conf, class

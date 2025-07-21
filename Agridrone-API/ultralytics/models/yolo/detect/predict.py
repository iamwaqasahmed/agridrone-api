# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops

def print_tensor_shapes_in_tuple(data):
    import torch
    """
    Recursively traverse a tuple (which may include nested tuples and lists)
    and print the shape of each torch.Tensor it contains.
    
    Parameters:
    - data (tuple): A tuple that can contain torch tensors, lists, or further tuples.
    
    Returns:
    None
    """
    # Only process if the current element is a tuple or list
    if isinstance(data, (tuple, list)):
        for item in data:
            if isinstance(item, torch.Tensor):
                # Print the shape of the tensor.
                print(item.shape)
            elif isinstance(item, (tuple, list)):
                # Recursively search through nested tuples or lists.
                print_tensor_shapes_in_tuple(item)
            # Other types are safely ignored.
    elif isinstance(data, torch.Tensor):
        print(data.shape)
    else:
        # If data is not a tuple or list, do nothing.
        return 
    
class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        print('***** Calling def postprocess() in yolo/detect/predict.py ******')
        # print()
        print(f'preds in yolo/detect/predict.py before nms: type(preds) {type(preds)}')
        print(f'preds shapes before nms')
        print_tensor_shapes_in_tuple(preds)
        # assert 0
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        print(f'preds shapes after nms')
        print_tensor_shapes_in_tuple(preds)
        
        print('the output seems to be num_classes * a number, 38 for 6 classes')
        # assert 0
        print(f'****** Calling: self.construct_results in yolo/detect/predict.py ******')
        return self.construct_results(preds, img, orig_imgs, **kwargs)

    def construct_results(self, preds, img, orig_imgs):
        """
        Constructs a list of result objects from the predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes and scores.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list): List of result objects containing the original images, image paths, class names, and bounding boxes.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Constructs the result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes and scores.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.

        Returns:
            (Results): The result object containing the original image, image path, class names, and bounding boxes.
        """
        # print(f'in detect/predict.py \npred: {pred}')
        # assert 0
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])

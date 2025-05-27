import torch
class SegmentationMetrics:
    def __init__(self, threshold=0.5, eps=1e-7):
        self.threshold = threshold
        self.eps = eps

    def _binarize(self, pred, target):
        pred = (pred > self.threshold).float()
        target = (target > self.threshold).float()
        return pred, target

    def pixel_accuracy(self, pred, target):
        pred, target = self._binarize(pred, target)
        correct = (pred == target).float().sum()
        total = torch.numel(pred)
        return (correct / (total + self.eps)).item()

    def mean_pixel_accuracy(self, pred, target):
        pred, target = self._binarize(pred, target)
        classes = [0, 1]
        accs = []
        for c in classes:
            mask = (target == c)
            if mask.sum() == 0:
                continue
            acc = ((pred == c) * mask).sum() / (mask.sum() + self.eps)
            accs.append(acc)
        if len(accs) == 0:
            return 0.0
        return torch.stack(accs).mean().item()

    def iou(self, pred, target):
        pred, target = self._binarize(pred, target)
        inter = (pred * target).sum(dim=(1,2,3))
        union = ((pred + target) > 0).float().sum(dim=(1,2,3))
        iou = (inter + self.eps) / (union + self.eps)
        return iou.mean().item()

    def precision(self, pred, target):
        pred, target = self._binarize(pred, target)
        tp = (pred * target).sum(dim=(1,2,3))
        fp = (pred * (1 - target)).sum(dim=(1,2,3))
        precision = (tp + self.eps) / (tp + fp + self.eps)
        return precision.mean().item()

    def recall(self, pred, target):
        pred, target = self._binarize(pred, target)
        tp = (pred * target).sum(dim=(1,2,3))
        fn = ((1 - pred) * target).sum(dim=(1,2,3))
        recall = (tp + self.eps) / (tp + fn + self.eps)
        return recall.mean().item()

    def f1_score(self, pred, target):
        p = self.precision(pred, target)
        r = self.recall(pred, target)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r + self.eps)

    def dice_coef(self, pred, target):
        pred, target = self._binarize(pred, target)
        inter = (pred * target).sum(dim=(1,2,3))
        union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = (2 * inter + self.eps) / (union + self.eps)
        return dice.mean().item()

    def evaluate(self, pred, target):
        return {
            "pixel_accuracy": self.pixel_accuracy(pred, target),
            "mean_pixel_accuracy": self.mean_pixel_accuracy(pred, target),
            "iou": self.iou(pred, target),
            "precision": self.precision(pred, target),
            "recall": self.recall(pred, target),
            "f1_score": self.f1_score(pred, target),
            "dice_coef": self.dice_coef(pred, target)
        }

if __name__ == "__main__":
    metrics = SegmentationMetrics()
    pred = torch.rand(1, 1, 100, 100)
    target = torch.rand(1, 1, 100, 100)
    print(metrics.evaluate(pred, target))
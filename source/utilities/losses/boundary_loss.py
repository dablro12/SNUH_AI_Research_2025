import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def safe_one_hot(label, n_classes):
    # label: (N, H, W), n_classes: int
    # label 값이 0~n_classes-1 범위 내에 있는지 체크
    label = label.clone()
    label[label < 0] = 0
    label[label >= n_classes] = n_classes - 1
    return F.one_hot(label, num_classes=n_classes)

class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        pred: (N, C, H, W) - logits or probabilities
        gt: (N, 1, H, W) or (N, H, W) - 각 픽셀의 클래스 인덱스
        """
        n, c, h, w = pred.shape

        # gt shape이 (N, 1, H, W)면 squeeze
        if gt.dim() == 4 and gt.size(1) == 1:
            gt = gt.squeeze(1)  # (N, H, W)

        # gt가 float(0.0, 1.0)이면 int로 변환
        if gt.dtype != torch.long:
            gt = gt.long()
        # 0,1 이외 값이 들어올 수 있으니, 0~c-1 범위로 클램프
        gt = torch.clamp(gt, 0, c-1)

        # softmax
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth (N, H, W) -> (N, H, W, C)
        one_hot_gt = safe_one_hot(gt, n_classes=c)  # (N, H, W, C)
        one_hot_gt = one_hot_gt.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)
        BF1 = 2 * P * R / (P + R + 1e-7)
        loss = torch.mean(1 - BF1)
        return loss

# for debug
if __name__ == "__main__":
    import torch.optim as optim
    from torchvision.models import segmentation

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = torch.randn(8, 3, 224, 224).to(device)
    gt = torch.randint(0, 10, (8, 224, 224)).to(device)
    # gt에 음수나 c이상 값이 들어가면 에러 발생하므로, 안전하게 0~c-1로만 생성

    model = segmentation.fcn_resnet50(num_classes=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = BoundaryLoss()

    y = model(img)

    loss = criterion(y['out'], gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)
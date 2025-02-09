import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor

# Load MiDaS model (Fastest Version)
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

# Load and preprocess image
image = cv2.imread("capture.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transform = Compose([ToTensor()])
image = transform(image).unsqueeze(0)

# Run inference
with torch.no_grad():
    depth = model(image)

# Convert depth map to numpy
depth_map = depth.squeeze().cpu().numpy()
depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Show depth map
cv2.imshow("Depth Map", depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

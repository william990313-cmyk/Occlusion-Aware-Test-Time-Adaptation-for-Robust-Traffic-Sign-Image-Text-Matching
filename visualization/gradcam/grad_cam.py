
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_image, target_class=None):
        self.model.zero_grad()

        # Forward pass
        image_embeddings = self.model.get_image_embeddings(input_image)

        if target_class is None:
            # If no target class is specified, use the class with the highest score
            # For image-text matching, we might need to define a 'target' based on similarity
            # For simplicity, let's assume we want to visualize what contributes to the image embedding itself
            # We can backpropagate from a specific dimension of the embedding or the whole embedding
            # Here, we'll backpropagate from the sum of the embedding dimensions.
            # This is a simplification for visualization purposes in a non-classification context.
            loss = image_embeddings.sum()
        else:
            # If it's a classification task, you'd typically use model(input_image)[0, target_class].backward()
            # For image-text matching, this part needs careful consideration based on how 'target_class' is defined.
            # For now, we'll stick to the sum of embeddings for heatmap generation on the image side.
            loss = image_embeddings[0, target_class]

        loss.backward()

        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Pool the gradients across the channels
        weights = np.mean(gradients, axis=(1, 2))

        # Weighted combination of activations
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # ReLU on heatmap
        heatmap = np.maximum(heatmap, 0)

        # Normalize the heatmap to [0, 1]
        heatmap /= np.max(heatmap) + 1e-12

        return heatmap

def show_cam_on_image(img, heatmap):
    # img is a torch tensor, convert to numpy and denormalize
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img



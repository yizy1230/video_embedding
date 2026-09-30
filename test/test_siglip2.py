import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

# load the model and processor
# ckpt = "google/siglip2-base-patch16-384"
ckpt = "./checkpoints/siglip2-so400m-patch16-384"
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)

print('loaded model and processor')
print('model device:', model.device)

# load the image
image = load_image("https://ark-project.tos-cn-beijing.volces.com/doc_image/ark_demo_img_1.png")
inputs = processor(images=[image], return_tensors="pt").to(model.device)

# run infernece
with torch.no_grad():
    # image_embeddings = model.get_image_features(**inputs).pooler_output
    # norm = torch.norm(image_embeddings, p=2, dim=-1)
    # image_embeddings = image_embeddings / norm.unsqueeze(-1)
    image_embeddings = model.get_image_features(**inputs)
    image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
print(torch.sum(image_embeddings ** 2, dim=-1))
print(image_embeddings)
print(image_embeddings.shape)
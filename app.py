import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. تصميم واجهة الموقع
st.title("🛡️ نظام كشف عيوب التصنيع")
st.write("ارفع صورة الزجاجة لفحصها ومعرفة مكان العيب")

# 2. تحميل الموديل وتجهيزه (نفس شكل الموديل الأصلي)
def get_model():
    model = models.resnet34()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    # تحميل الأوزان التي حفظناها قبل قليل
    model.load_state_dict(torch.cuda.get_state_dict('model_trained.pth', map_location='cpu'))
    model.eval()
    return model

model = get_model()

# 3. خانة رفع الصورة
file = st.file_uploader("اختر صورة زجاجة من جهازك", type=["jpg", "png"])

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, caption="الصورة التي رفعتها", width=300)
    
    # تحويل الصورة لتناسب الموديل
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)

    # التنبؤ (Prediction)
    output = model(input_tensor)
    _, pred = torch.max(output, 1)
    
    # 4. عرض النتيجة
    if pred.item() == 0:
        st.error("النتيجة: زجاجة مَعيبة (Defective) ❌")
    else:
        st.success("النتيجة: زجاجة سليمة (Good) ✅")

    # 5. تشغيل Grad-CAM (شرح السبب)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred.item())])[0, :]
    
    img_np = np.array(img.resize((224, 224))) / 255.0
    vis = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    st.image(vis, caption="لماذا قرر الموديل ذلك؟ (الخريطة الحرارية)", width=300)

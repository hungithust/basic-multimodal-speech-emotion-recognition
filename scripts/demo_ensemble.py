#!/usr/bin/env python3
"""
Demo script cho Ensemble Model Inference
Hướng dẫn sử dụng ensemble model để predict emotion
"""

import torch
from inference.inference import (
    load_inference_models,
    create_ensemble_model,
    predict_emotion_from_wav,
    demo_ensemble_inference
)

def main():
    """
    Demo sử dụng ensemble model
    """
    print("🎯 Multimodal Speech Emotion Recognition - Ensemble Demo")
    print("=" * 60)

    # 1. Load các models cơ bản
    print("📦 Loading base models...")
    text_model, audio_model, fusion_model, tokenizer = load_inference_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Models loaded on device: {device}")

    # 2. Tạo ensemble model
    print("\n🔧 Creating ensemble model...")
    ensemble_model = create_ensemble_model(text_model, audio_model, device)
    print("✅ Ensemble model created!")
    print(f"   - Number of models: {ensemble_model.num_models}")
    print(f"   - Model weights: {ensemble_model.get_model_weights()}")

    # 3. Demo prediction (nếu có file WAV)
    print("\n🎤 Inference Demo:")
    print("Để test với file WAV thực tế, sử dụng code sau:")
    print("""
    # Single model prediction
    result_single = predict_emotion_from_wav(
        "path/to/audio.wav",
        text_model, audio_model, fusion_model, tokenizer,
        use_ensemble=False
    )

    # Ensemble prediction
    result_ensemble = predict_emotion_from_wav(
        "path/to/audio.wav",
        text_model, audio_model, fusion_model, tokenizer,
        use_ensemble=True, ensemble_model=ensemble_model
    )

    print(f"Single: {result_single['predicted_emotion']}")
    print(f"Ensemble: {result_ensemble['predicted_emotion']}")
    """)

    # 4. Thông tin về ensemble
    print("\n📊 Ensemble Information:")
    print("- FusionModel: Kết hợp features + classification outputs")
    print("- EarlyFusionModel: Kết hợp ở lớp fully connected thứ nhất (256)")
    print("- LateFusionModel: Kết hợp ở lớp fully connected thứ hai (128)")
    print("- Weights dựa trên accuracy validation (có thể cập nhật)")

    print("\n✨ Ensemble model ready for production use!")

if __name__ == "__main__":
    main()
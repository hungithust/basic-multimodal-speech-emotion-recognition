import torch  
import numpy as np  
import pandas as pd  
from transformers import AutoTokenizer  
import os

  
from preprocessing.iemocap import IemocapPreprocessor  
from audio.extractor import Wav2Vec2Extractor  
from audio.wav2vec2 import Wav2Vec2  
from text.deberta import DebertaV3, DebertaV3Tokenizer  
from fusion.model import FusionModel, EarlyFusionModel, LateFusionModel, EnsembleModel  
from core.config import CONFIG  
  
def load_inference_models():  
    """Tải tất cả models đã train cho inference"""  
    # Load configuration  
    CONFIG.load_config("config.yaml")  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    # Load text model và tokenizer  
    # text_model = DebertaV3(num_classes=len(CONFIG.dataset_emotions()))  
    text_model = torch.load(
        os.path.join(CONFIG.saved_models_location(), 'deberta_model3.pt'),
        map_location='cuda',
    )
    
    text_tokenizer = DebertaV3Tokenizer()  
      
    # Load audio model  
    audio_model = Wav2Vec2(num_classes=len(CONFIG.dataset_emotions()))  
      
    # Load fusion model  
    fusion_model = FusionModel(  
        num_classes=len(CONFIG.dataset_emotions()),  
        deberta_model=text_model,  
        wav2vec2_model=audio_model  
    )  
      
    # Load trained weights (cần có file weights)  
    # text_model.load_state_dict(torch.load("saved_models/deberta_model3.pt"))  
    audio_model.load_state_dict(torch.load("saved_models/wav2vec2_state_dict3.pt"))  
    fusion_model.load_state_dict(torch.load("saved_models/fusion_state_dict.pt"))  
    
    text_model.to(device)  
    audio_model.to(device)  # QUAN TRỌNG: Audio model cũng cần lên GPU  
    fusion_model.to(device)  


    # Set to evaluation mode  
    text_model.eval()  
    audio_model.eval()  
    fusion_model.eval()  
      
    return text_model, audio_model, fusion_model, text_tokenizer  


def create_ensemble_model(text_model, audio_model, device):
    """
    Tạo ensemble model kết hợp nhiều fusion strategies
    Sử dụng weights dựa trên accuracy (tạm thời fix cứng)
    """
    num_classes = len(CONFIG.dataset_emotions())

    # Tạo các fusion models khác nhau
    fusion_model = FusionModel(num_classes, text_model, audio_model)
    early_fusion_model = EarlyFusionModel(num_classes, text_model, audio_model)
    late_fusion_model = LateFusionModel(num_classes, text_model, audio_model)

    # Load weights cho các models (giả sử có file)
    try:
        fusion_model.load_state_dict(torch.load("saved_models/fusion_state_dict.pt"))
        early_fusion_model.load_state_dict(torch.load("saved_models/early_fusion_state_dict.pt"))
        late_fusion_model.load_state_dict(torch.load("saved_models/late_fusion_state_dict.pt"))
    except FileNotFoundError:
        print("Warning: Some model weights not found, using untrained models")

    # Chuyển models lên device
    fusion_model.to(device)
    early_fusion_model.to(device)
    late_fusion_model.to(device)

    # Set evaluation mode
    fusion_model.eval()
    early_fusion_model.eval()
    late_fusion_model.eval()

    # Weights dựa trên accuracy (tạm thời fix cứng, có thể thay đổi sau)
    # Giả sử: FusionModel: 0.85, EarlyFusion: 0.82, LateFusion: 0.83
    ensemble_weights = [0.85, 0.82, 0.83]

    # Tạo ensemble model
    ensemble_model = EnsembleModel(
        fusion_models=[fusion_model, early_fusion_model, late_fusion_model],
        weights=ensemble_weights,
        num_classes=num_classes
    )

    ensemble_model.to(device)
    ensemble_model.eval()

    return ensemble_model  
  
def predict_emotion_from_wav(wav_path: str, text_model, audio_model, fusion_model, tokenizer, use_ensemble: bool = False, ensemble_model=None):  
    """Predict emotion từ một file WAV"""  
      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # 1. Transcribe audio sang text  
    preprocessor = IemocapPreprocessor("")  # dataset_path không cần cho inference  
    transcription_list = preprocessor.batch_transcribe([wav_path])  # Trả về list  
    transcription = transcription_list[0]  # Lấy text đầu tiên  
    # 2. Extract audio features  
    extractor = Wav2Vec2Extractor()  
    audio_features = extractor.extract(wav_path)  
      
    # 3. Tokenize text  
    text_tokens = tokenizer.encode(  
        transcription,  
        add_special_tokens=True,  
        truncation=True,  
        padding="max_length",  
        max_length=256,  
        return_tensors="pt"  
    )  
      
    # 4. Prepare inputs  
    audio_input = torch.tensor(audio_features).unsqueeze(0).to(device) # Add batch dimension  
    print(text_tokens)
    text_input = text_tokens.to(device)
    print(text_input)
    # 5. Run inference  
    with torch.no_grad():  
        # Get individual model outputs  
        text_output = text_model(text_input)  
        text_logits = text_output.logits 
        audio_logits = audio_model(audio_input)  
          
        # Get fusion prediction  
        if use_ensemble and ensemble_model is not None:
            # Sử dụng ensemble model
            fusion_logits = ensemble_model(text_input, audio_input)
            model_type = "ensemble"
        else:
            # Sử dụng single fusion model
            fusion_logits = fusion_model(text_input, audio_input)
            model_type = "single_fusion"
          
        # Get predicted emotion  
        predicted_class = torch.argmax(fusion_logits, dim=1).item()  
        emotions = CONFIG.dataset_emotions()  
        predicted_emotion = emotions[predicted_class]  
          
        # Get confidence scores  
        confidence_scores = torch.softmax(fusion_logits, dim=1).squeeze().tolist()  
      
    return {  
        "transcription": transcription,  
        "predicted_emotion": predicted_emotion,  
        "confidence_scores": dict(zip(emotions, confidence_scores)),  
        "text_logits": text_logits.squeeze().tolist(),  
        "audio_logits": audio_logits.squeeze().tolist(),
        "model_type": model_type
    }  
  
def batch_predict_emotions(wav_paths: list, text_model, audio_model, fusion_model, tokenizer, use_ensemble: bool = False, ensemble_model=None):  
    """Predict emotion cho batch file WAV"""  
    results = []  
      
    for wav_path in wav_paths:  
        result = predict_emotion_from_wav(  
            wav_path, text_model, audio_model, fusion_model, tokenizer, use_ensemble, ensemble_model  
        )  
        result["audio_path"] = wav_path  
        results.append(result)  
      
    return results


def demo_ensemble_inference():
    """
    Demo function để show cách sử dụng ensemble model
    """
    print("=== Demo Ensemble Inference ===")

    # Load models
    text_model, audio_model, fusion_model, tokenizer = load_inference_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tạo ensemble model
    ensemble_model = create_ensemble_model(text_model, audio_model, device)

    # Hiển thị thông tin ensemble
    print(f"Ensemble weights: {ensemble_model.get_model_weights()}")
    print(f"Number of models in ensemble: {ensemble_model.num_models}")

    # Giả sử có file WAV để test
    # test_wav = "path/to/test.wav"

    # So sánh single model vs ensemble
    # single_result = predict_emotion_from_wav(test_wav, text_model, audio_model, fusion_model, tokenizer, use_ensemble=False)
    # ensemble_result = predict_emotion_from_wav(test_wav, text_model, audio_model, fusion_model, tokenizer, use_ensemble=True, ensemble_model=ensemble_model)

    # print(f"Single model prediction: {single_result['predicted_emotion']}")
    # print(f"Ensemble prediction: {ensemble_result['predicted_emotion']}")

    print("Ensemble model ready for inference!")
import torch  
import numpy as np  
import pandas as pd  
from transformers import AutoTokenizer  
import os

  
from preprocessing.iemocap import IemocapPreprocessor  
from audio.extractor import Wav2Vec2Extractor  
from audio.wav2vec2 import Wav2Vec2  
from text.deberta import DebertaV3, DebertaV3Tokenizer  
from fusion.model import FusionModel  
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
  
def predict_emotion_from_wav(wav_path: str, text_model, audio_model, fusion_model, tokenizer):  
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
        fusion_logits = fusion_model(text_input, audio_input)  
          
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
        "audio_logits": audio_logits.squeeze().tolist()  
    }  
  
def batch_predict_emotions(wav_paths: list, text_model, audio_model, fusion_model, tokenizer):  
    """Predict emotion cho batch file WAV"""  
    results = []  
      
    for wav_path in wav_paths:  
        result = predict_emotion_from_wav(  
            wav_path, text_model, audio_model, fusion_model, tokenizer  
        )  
        result["audio_path"] = wav_path  
        results.append(result)  
      
    return results
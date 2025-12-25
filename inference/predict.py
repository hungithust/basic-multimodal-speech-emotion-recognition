import torch  
import pandas as pd  
import numpy as np  
import random
from audio.extractor import Wav2Vec2Extractor  
from text.deberta import DebertaV3Tokenizer, DebertaV3



from scripts.run_model import TrainerOps  
from core.config import CONFIG, device  
from scripts.preprocess_data import (  
    process_audio_data_to_pickle,
    process_raw_data_to_pickle,
    process_text_data_to_pickle
)

from scripts.get_dataloaders import get_dataloader  


def predict_emotions_from_folder(inference_folder: str):  
    """Run inference on all WAV files in a folder"""  
    def prepare_env():
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        CONFIG.load_config("config.yaml")
    prepare_env()
    # 1. Generate inference dataframe  
    # 1. Prepare audio paths and raw text
    process_raw_data_to_pickle("audio_and_text.pkl", inference=True)
    # 2. Turn the raw audio file names into mfccs
    process_audio_data_to_pickle(
        "audio_and_text.pkl", "w2v2_and_text.pkl", Wav2Vec2Extractor(), inference=True
    )
    # 3. Turn the raw text file into tokens
    process_text_data_to_pickle(
        "w2v2_and_text.pkl", "w2v2_and_tokens.parquet", DebertaV3Tokenizer(), inference=True
    )

    text_trainer = TrainerOps.create_or_load_text_trainer("deberta_model3.pt")
    audio_trainer = TrainerOps.create_or_load_audio_trainer("wav2vec2_state_dict3.pt", load_state_dict=True) 
    # 4. Load trained model  
    fusion_trainer = TrainerOps.create_or_load_fusion_trainer(  
        load_path="fusion_state_dict.pt",   
        audio_model=audio_trainer.model,
        text_model=text_trainer.model,  
        load_state_dict=True
    )  
      
    # 5. Get inference dataloader  

    inference_loader = get_dataloader(inference=True)  
      
    # 6. Run predictions  
    predictions = []  
    fusion_trainer.model.eval()  
    with torch.no_grad():  
        for batch in inference_loader:  
            audio, text = batch  
            audio = audio.to(device)  
            text = text.to(device)  
              
            outputs = fusion_trainer.model(text, audio)  
            pred_indices = torch.argmax(outputs, dim=1)  
            predictions.extend(pred_indices.cpu().numpy())  
      
    # 7. Convert to emotion labels  
    emotions = CONFIG.dataset_emotions()  
    results = [emotions[idx] for idx in predictions]  
      
    return results
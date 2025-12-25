# import re
# import os

# import pandas as pd


# class IemocapPreprocessor:
#     def __init__(self, dataset_path: str):
#         self._dataset_path = dataset_path

#         # self._info_line = re.compile(r"\[.+\]\n", re.IGNORECASE)
        
        
#     # bien doi tu wav ve text
#     # 1. dua qua phowhisper de phien am
#     # 2. dua qua photonx de sua loi
#     def transcript_data(self, audio_path: str) -> str:
#         import torch
#         from transformers import WhisperProcessor, WhisperForConditionalGeneration
#         import librosa
#         import numpy as np

#         # 1. Định nghĩa tên mô hình PhoWhisper Small
#         model_name = "vinai/phowhisper-small"

#         # 2. Tải Processor và Model
#         # Processor sẽ tải tokenizer và feature extractor
#         processor = WhisperProcessor.from_pretrained(model_name)
#         model = WhisperForConditionalGeneration.from_pretrained(model_name)

#         # 3. Chuyển mô hình sang GPU nếu có
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model.to(device) # type: ignore

#         # print(f"Đã tải mô hình {model_name} và sử dụng trên thiết bị: {device}")
        
#         def transcribe_vietnamese_audio(audio_path):
#             """
#             Chuyển đổi tệp âm thanh Tiếng Việt sang văn bản bằng PhoWhisper Small.

#             Args:
#                 audio_path (str): Đường dẫn đến tệp âm thanh (ví dụ: .wav, .mp3).

#             Returns:
#                 str: Văn bản đã được chuyển đổi.
#             """
#             # 1. Tải và lấy mẫu lại (resample) tệp âm thanh về tần số 16kHz (yêu cầu của Whisper)
#             # librosa.load trả về numpy array và sample rate (sr)
#             speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

#             # 2. Xử lý đầu vào: trích xuất features
#             # Cần đảm bảo rằng audio là một mảng 1D (mono)
#             input_features = processor(
#                 speech_array, 
#                 sampling_rate=sampling_rate, 
#                 return_tensors="pt"
#             ).input_features
            
#             # 3. Chuyển input features sang thiết bị (GPU/CPU)
#             input_features = input_features.to(device)

#             # 4. Chạy Inference (tạo văn bản)
#             with torch.no_grad():
#                 # Tham số language="vi" và task="transcribe" là bắt buộc đối với PhoWhisper
#                 predicted_ids = model.generate(
#                     input_features, 
#                     forced_decoder_ids=processor.get_decoder_prompt_ids(language="vi", task="transcribe")
#                 )

#             # 5. Giải mã ID thành chuỗi văn bản
#             transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]
            
#             return transcription
        
#         # def correct_transcription_with_photonx(transcription: str) -> str:
            
#         #     # phải cài pip install --upgrade photonx
#         #     from photonx import PhotonX
            
#         #     import os
#         #     os.environ["PROTONX_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im5ndXllbnZpZXRodW5nc29pY3RodXN0QGdtYWlsLmNvbSIsImlhdCI6MTc2NTY3Nzg0NywiZXhwIjoxNzY4MjY5ODQ3fQ.azkJoKM3TvwAsu_4NmPRifs8f9Vr8IXzMLP_GArNNxg"

#         #     # Khởi tạo PhotonX với mô hình Tiếng Việt
#         #     photonx = PhotonX()

#         #     # Sử dụng PhotonX để sửa lỗi chính tả và ngữ pháp
#         #     corrected_text = photonx.text.correct(input=transcription,top_k=1)['data'][0]['candidates'][0]['output']

#         #     return corrected_text
        
#         transcription = transcribe_vietnamese_audio(audio_path)
#         # corrected_transcription = correct_transcription_with_photonx(transcription)
#         corrected_transcription = transcription
#         return corrected_transcription
        
    
#     def labeling_data(self,folder_path:str, label:str) -> pd.DataFrame:
#         import os
#         import pandas as pd
#         from typing import List
        
#         """
#         Quét qua thư mục, gán nhãn cố định cho tất cả các tệp WAV, 
#         và trả về một DataFrame với cột 'audio' (tên tệp không đuôi) và 'emotion'.

#         Args:
#             folder_path (str): Đường dẫn đến thư mục chứa tệp WAV.
#             emotion_label (str): Nhãn cảm xúc (ví dụ: "anger") muốn gán.

#         Returns:
#             pd.DataFrame: DataFrame với hai cột ('audio', 'emotion').
#         """
        
#         if not os.path.isdir(folder_path):
#             print(f"Lỗi: Thư mục không tồn tại tại đường dẫn: {folder_path}")
#             return pd.DataFrame()

#         audio_names: List[str] = [] # Danh sách lưu tên tệp không có đuôi

#         for file_name in os.listdir(folder_path):
#             # 1. Kiểm tra và lọc tệp WAV (không phân biệt chữ hoa/thường)
#             if file_name.lower().endswith('.wav'):
#                 # 2. Xóa đuôi .wav khỏi tên tệp
#                 name_without_extension = file_name[:-4] 
#                 # Giả sử file luôn kết thúc bằng .wav (4 ký tự)
#                 audio_names.append(folder_path + name_without_extension)

#         if not audio_names:
#             print(f"Cảnh báo: Không tìm thấy tệp .wav nào trong thư mục: {folder_path}")
#             return pd.DataFrame()
            
#         # 3. Tạo DataFrame với tên cột mới
#         data = {
#             'audio': audio_names,
#             'emotion': [label] * len(audio_names) # Gán nhãn cố định
#         }
        
#         df = pd.DataFrame(data)
#         print(f"Đã tạo DataFrame với {len(df)} hàng.")
#         return df
        



#     def generate_dataframe(self) -> pd.DataFrame:
#         audios = []
#         emotions = []
#         texts = []

    
#         # read folder
#         sca_folder = os.path.join(self._dataset_path, "scare") #scare
#         sup_folder = os.path.join(self._dataset_path, "surprise") #surprise
        
#         # process labeling data
#         sca_df = self.labeling_data(sca_folder, "scare")
#         sup_df = self.labeling_data(sup_folder, "surprise")
        
        
#         # read metadata hap, neu, sad, ang
#         metadata_path = os.path.join(self._dataset_path,'happy+neutral+angry+sad' ,"metadata.csv")
#         metadata_df = pd.read_csv(metadata_path)
#         metadata_df['audio'] = metadata_df['filename'].str.replace('.wav', '', regex=False) # remove .wav extension
        
#         happy_neutral_angry_sad_path = os.path.join(self._dataset_path,'happy+neutral+angry+sad','converted' )
#         metadata_df['audio'] = happy_neutral_angry_sad_path + metadata_df['audio']
#         # metadata_df['audio'] = metadata_df['audio'].apply(lambda x: os.path.join(happy_neutral_angry_sad_path, x + '.wav'))
#         metadata_df = metadata_df[['audio','emotion']]
        
        
#         # combine all dataframe
#         data_df = pd.concat([sca_df, sup_df, metadata_df], ignore_index=True)
        
        
#         # process transcript data
#         for index, row in data_df.iterrows():
#             audio_path = row['audio'] + '.wav'  # Thêm đuôi .wav vào đường dẫn tệp âm thanh
#             text = self.transcript_data(audio_path)
#             audios.append(audio_path)
#             texts.append(text)
#             emotions.append(row['emotion'])
#         # convert to dataframe and shuffle data
#         df_shuffled = pd.DataFrame(data={"audio": audios, "text": texts, "emotion": emotions}).sample(frac=1, random_state=42).reset_index(drop=True)

        
        

#         return df_shuffled


import os
import pandas as pd
from typing import List
from tqdm import tqdm


class IemocapPreprocessor:
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 12
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        # ===== Load model 1 lần =====
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "vinai/phowhisper-small"

        print(f"[INFO] Loading model {self.model_name} on {self.device}")

        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        print("[INFO] Model loaded successfully")
        
    def labeling_data(self, folder_path: str, label: str) -> pd.DataFrame:
        audio_paths = []

        for f in os.listdir(folder_path):
            if f.lower().endswith(".wav"):
                audio_paths.append(
                    os.path.join(folder_path, f)  # bỏ .wav
                )

        return pd.DataFrame({
            "audio": audio_paths,
            "emotion": label
        })

    # def batch_transcribe(self, audio_paths: List[str]) -> List[str]:
    #     import torch
    #     import torchaudio
    #     from torch.amp import autocast

    #     results = []

    #     forced_decoder_ids = self.processor.get_decoder_prompt_ids(
    #         language="vi",
    #         task="transcribe"
    #     )

    #     for i in tqdm(range(0, len(audio_paths), self.batch_size), desc="Transcribing"):
    #         batch_paths = audio_paths[i:i + self.batch_size]
    #         speeches = []

    #         for path in batch_paths:
    #             waveform, sr = torchaudio.load(path)
    #             waveform = waveform.mean(dim=0)  # mono
    #             if sr != 16000:
    #                 waveform = torchaudio.functional.resample(
    #                     waveform, sr, 16000
    #                 )
    #             speeches.append(waveform.numpy())

    #         inputs = self.processor(
    #             speeches,
    #             sampling_rate=16000,
    #             return_tensors="pt",
    #             # padding=True
    #             padding="max_length",
    #             max_length=1500,
    #             truncation=True
    #         ).input_features.to(self.device)

    #         with torch.no_grad():
    #             with autocast(device_type="cuda", enabled=self.device == "cuda"):
    #                 predicted_ids = self.model.generate(
    #                     inputs,
    #                     forced_decoder_ids=forced_decoder_ids
    #                 )

    #         texts = self.processor.batch_decode(
    #             predicted_ids,
    #             skip_special_tokens=True
    #         )

    #         results.extend(texts)

    #     return results
    
    def batch_transcribe(self, audio_paths):
        import torch
        import torchaudio
        import torch.nn.functional as F
        from torch.cuda.amp import autocast
        from tqdm import tqdm

        results = []

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="vi",
            task="transcribe"
        )

        MAX_MEL_LEN = 3000  # 🔥 BẮT BUỘC

        for i in tqdm(range(0, len(audio_paths), self.batch_size), desc="Transcribing"):
            batch_paths = audio_paths[i:i + self.batch_size]
            mel_batch = []

            for path in batch_paths:
                waveform, sr = torchaudio.load(path)
                waveform = waveform.mean(dim=0)

                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)

                # 🔥 TẠO MEL SPECTROGRAM
                mel = self.processor.feature_extractor(
                    waveform.numpy(),
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features[0]   # (80, T)

                # 🔥 PAD / TRUNCATE MEL → 3000
                T = mel.shape[1]
                if T < MAX_MEL_LEN:
                    mel = F.pad(mel, (0, MAX_MEL_LEN - T))
                else:
                    mel = mel[:, :MAX_MEL_LEN]

                mel_batch.append(mel)

            inputs = torch.stack(mel_batch).to(self.device)  # (B, 80, 3000)

            # 🔒 ASSERT CHỐNG CRASH
            assert inputs.shape[-1] == 3000, inputs.shape

            with torch.no_grad():
                with autocast(enabled=self.device == "cuda"):
                    predicted_ids = self.model.generate(
                        input_features=inputs,
                        forced_decoder_ids=forced_decoder_ids
                    )

            texts = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )

            results.extend(texts)

        return results

    def generate_dataframe(self) -> pd.DataFrame:
        # ===== scare & surprise =====
        sca_folder = os.path.join(self.dataset_path, "scare")
        sup_folder = os.path.join(self.dataset_path, "surprise")

        sca_df = self.labeling_data(sca_folder, "scare")
        sup_df = self.labeling_data(sup_folder, "surprise")

        # ===== happy + neutral + angry + sad =====
        metadata_path = os.path.join(
            self.dataset_path,
            "happy+neutral+angry+sad",
            "metadata.csv"
        )

        meta_df = pd.read_csv(metadata_path)


        base_path = os.path.join(
            self.dataset_path,
            "happy+neutral+angry+sad",
            "converted"
        )

        meta_df["audio"] = meta_df["filename"].apply(
            lambda x: os.path.join(base_path, x)
        )

        meta_df = meta_df[["audio", "emotion"]]

        # ===== Combine =====
        data_df = pd.concat(
            [sca_df, sup_df, meta_df],
            ignore_index=True
        )

        print(f"[INFO] Total audio files: {len(data_df)}")

        # ===== Transcribe =====
        texts = self.batch_transcribe(data_df["audio"].tolist())

        data_df["text"] = texts

        # ===== Shuffle =====
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)


        return data_df

    # inference
    def generate_inference_dataframe(self) -> pd.DataFrame:
        import glob

        audio_paths = glob.glob(
            os.path.join(self.inference_dataset_path, ".wav")
        )

        print(f"[INFO] Total inference audio files: {len(audio_paths)}")

        texts = self.batch_transcribe(audio_paths)

        data_df = pd.DataFrame({
            "audio": audio_paths,
            "text": texts
        })

        return data_df
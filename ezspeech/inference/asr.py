import torch
from hydra.utils import instantiate
from ezspeech.data.util import extract_audio_feature
import torchaudio
from ezspeech.utils.asr import load_vocab
import json
from torch.nn.utils.rnn import pad_sequence
from pyctcdecode import build_ctcdecoder
import multiprocessing


class EzASR:
    def __init__(self,filepath,vocab_file,device="cuda",blank_idx=0):
        self.blank_idx=blank_idx
        self.device=device
        self.encoder=self._load_module(filepath).to(self.device)
        self.encoder=self.encoder.eval()
        self.vocab=open(vocab_file).read().splitlines()
        self.beam_seach_decoder = build_ctcdecoder(
                    self.vocab,
                    kenlm_model_path="/home4/khanhnd/Ezspeech/3-gram.pruned.1e-7.arpa",  # either .arpa or .bin file
                    alpha=0.5,  # tuned on a val set
                    beta=1.0,  # tuned on a val set
                )

    
    def _load_module(self,filepath):
        checkpoint=torch.load(filepath,weights_only=False)
        encoder=instantiate(checkpoint["hyper_parameters"]["encoder"])
        encoder.load_state_dict(checkpoint["state_dict"]["encoder"])
        return encoder
    def feature_extraction(self, filepaths: list[str]):
        wavs=[torchaudio.load(i)[0] for i in filepaths]
        sample_rates=[torchaudio.load(i)[1] for i in filepaths]
        assert sample_rates[0]==16000
        
        
        audio_features=[extract_audio_feature(i,j).to(self.device).t() for i,j in zip(wavs,sample_rates)]
        audio_feature_length = torch.tensor([len(i) for i in audio_features],dtype=torch.long)
        audio_features = pad_sequence(audio_features, batch_first=True)

        return audio_features.to(self.device),audio_feature_length.to(self.device)
    def encode(self,audio_features,audio_feature_length ):
        logits,lengths=self.encoder(audio_features,audio_feature_length)
        return logits.detach()
    def greedy_decode(self,logits:list[torch.Tensor]):
        
        labels = torch.argmax(logits, dim=-1)
        labels_no_blank=[torch.unique_consecutive(label[label != self.blank_idx]) for label in labels]
        transcript_lst=[]
        for predict in labels_no_blank:
            transcript=[self.vocab[i] for i in predict]
            transcript="".join(transcript)
            transcript=transcript.replace("|"," ")
            transcript_lst.append(transcript)
        return transcript_lst
    def beam_search_decode(self,logits: list[torch.Tensor]):
        logits=logits.cpu().numpy()

        with multiprocessing.get_context("fork").Pool() as pool:
            text_list = self.beam_seach_decoder.decode_batch(pool, logits)
        # text_list=self.beam_seach_decoder.decode(logits[0])
        return text_list
    def greedy_transcribe(self,filepaths):
        audio_features,audio_feature_length=self.feature_extraction(filepaths)
        logits=self.encode(audio_features,audio_feature_length)
        transcript_lst=self.greedy_decode(logits)
        return transcript_lst
    def transcribe(self,filepaths):
        audio_features,audio_feature_length=self.feature_extraction(filepaths)
        logits=self.encode(audio_features,audio_feature_length)
        transcript_lst=self.beam_search_decode(logits)
        return transcript_lst
from jiwer import wer
from tqdm import tqdm
if __name__=="__main__":
    all_wer=[]
    a=EzASR("/home4/khanhnd/Ezspeech/exported_checkpoint/hehe.ckpt","/home4/khanhnd/Ezspeech/ezspeech/resources/vocab_en.txt")
    import pandas as pd
    df=pd.read_csv("/home4/khanhnd/Ezspeech/data/librispeech_train_100h.tsv",sep="\t")
    for idx,i in tqdm(df.iterrows(),total=len(df)):
        res=a.transcribe([i["audio_filepath"]])[0]
        print(res)
        all_wer.append(wer(i["transcript"],res))

    print(sum(all_wer)/len(df))



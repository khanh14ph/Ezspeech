import torch
from hydra.utils import instantiate
from ezspeech.data.util import extract_audio_feature
import torchaudio
from ezspeech.utils.asr import load_vocab
import json
from torch.nn.utils.rnn import pad_sequence
class EzASR:
    def __init__(self,filepath,vocab_file,device="cuda",blank_idx=0):
        self.blank_idx=blank_idx
        self.device=device
        self.encoder=self._load_module(filepath).to(self.device)
        self.vocab=load_vocab(vocab_file)

    
    def _load_module(self,filepath):
        checkpoint=torch.load(filepath)
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
        print("audio_feature_length",audio_feature_length)
        return audio_features.to(self.device),audio_feature_length.to(self.device)
    def encode(self,audio_features,audio_feature_length ):
        logits,lengths=self.encoder(audio_features,audio_feature_length)
        print("logits",logits.shape)
        return logits
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
    def transcribe(self,filepaths):
        audio_features,audio_feature_length=self.feature_extraction(filepaths)
        logits=self.encode(audio_features,audio_feature_length)
        transcript_lst=self.greedy_decode(logits)
        return transcript_lst

if __name__=="__main__":

    a=EzASR("/home4/khanhnd/Ezspeech/exported_checkpoint/hehe.ckpt","/home4/khanhnd/Ezspeech/ezspeech/resources/vocab.json")
    import pandas as pd
    df=pd.read_csv("/home4/khanhnd/vivos/train.tsv",sep="\t")
    lst=list(df[0:1]["audio_filepath"])
    res=a.transcribe(lst)
    print(res)



# For use within https://github.com/CorentinJ/Real-Time-Voice-Cloning
from encoder import inference as encoder
class EncoderServe(object):
    def __init__(path_weight=""):
        encoder.load_model(path_weight)
    def predict(x, feature_names):
        #preprocessed_wav = encoder.preprocess_wav(x)
        # - If the wav is already loaded:
        #original_wav, sampling_rate = librosa.load(x)
        #preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        #print("Loaded file succesfully")
        # Then we derive the embedding. There are many functions and parameters that the 
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        embed = encoder.embed_utterance(x)

EncoderServe()
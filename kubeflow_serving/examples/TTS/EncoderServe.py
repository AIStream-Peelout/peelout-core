# For use within https://github.com/CorentinJ/Real-Time-Voice-Cloning
from encoder.model import SpeakerEncoder

class EncoderServe(object):
    def __init__(path_weight="pretrained.pt"):
        print("the weight below")
        self.model = load_model(path_weight)
    def predict(x, feature_names):
        #preprocessed_wav = encoder.preprocess_wav(x)
        # - If the wav is already loaded:
        #original_wav, sampling_rate = librosa.load(x)
        #preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        #print("Loaded file succesfully")
        # Then we derive the embedding. There are many functions and parameters that the 
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        embed = embed_utterance(x, model)



def load_model(weights_fpath: str, device=None):
    """
    Loads the model in memory. If this function is not explicitely called, it will be run on the 
    first call to embed_frames() with the default weights file.
    
    :param weights_fpath: the path to saved model weights.
    :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). The 
    model will be loaded and will run on this device. Outputs will however always be on the cpu. 
    If None, will default to your GPU if it"s available, otherwise your CPU.
    """
    # TODO: I think the slow loading of the encoder might have something to do with the device it
    #   was saved on. Worth investigating.
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        _device = torch.device(device)
    model = SpeakerEncoder(_device, torch.device("cpu"))
    print(weights_fpath)
    checkpoint = torch.load(weights_fpath)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()
    print("Loaded encoder \"%s\" trained to step %d" % (weights_fpath.name, checkpoint["step"]))

def embed_utterance(wav, using_partials=True, return_partials=False, **kwargs):
    """
    Computes an embedding for a single utterance.
    
    # TODO: handle multiple wavs to benefit from batching on GPU
    :param wav: a preprocessed (see audio.py) utterance waveform as a numpy array of float32
    :param using_partials: if True, then the utterance is split in partial utterances of 
    <partial_utterance_n_frames> frames and the utterance embedding is computed from their 
    normalized average. If False, the utterance is instead computed from feeding the entire 
    spectogram to the network.
    :param return_partials: if True, the partial embeddings will also be returned along with the 
    wav slices that correspond to the partial embeddings.
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If 
    <return_partials> is True, the partial utterances as a numpy array of float32 of shape 
    (n_partials, model_embedding_size) and the wav partials as a list of slices will also be 
    returned. If <using_partials> is simultaneously set to False, both these values will be None 
    instead.
    """
    # Process the entire utterance if not using partials
    if not using_partials:
        frames = audio.wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed
    
    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    
    # Split the utterance into partials
    frames = audio.wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch)
    
    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)
    
    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed


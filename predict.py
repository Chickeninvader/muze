import torchaudio
from tqdm import trange
from audiocraft.models import MusicGen
import json
import torch

model = MusicGen.get_pretrained("facebook/musicgen-large")
model.lm.load_state_dict(torch.load('models/lm_final.pt'))
self = model
save_path = 'results/'
with open("private/private.json", "r") as f:
    data = json.load(f)
description_list = []
file_name_list = []
for file_name, description in data.items():
    description_list.append(description)
    file_name_list.append(file_name)
    if len(description_list) < 50:
        continue
    attributes, prompt_tokens = self._prepare_tokens_and_attributes(
        description_list, None)
    print("attributes:", attributes)
    print("prompt_tokens:", prompt_tokens)

    duration = 10
    self.generation_params = {
        'max_gen_len': int(duration * self.frame_rate),
        'use_sampling': 1,
        'temp': 1.0,
        'top_k': 250,
        'top_p': 0.0,
        'cfg_coef': 3.0,
        'two_step_cfg': 0,
    }
    total = []

    with self.autocast:
        gen_tokens = self.lm.generate(
            prompt_tokens, attributes, callback=None, **self.generation_params)
        total.append(gen_tokens[..., prompt_tokens.shape[-1]
                                if prompt_tokens is not None else 0:])
        prompt_tokens = gen_tokens[..., -gen_tokens.shape[-1] // 2:]
    gen_tokens = torch.cat(total, -1)

    with torch.no_grad():
        for i in range(2):
            if i == 0:
                save_path = 'results/submission1/'
            else:
                save_path = 'results/submission2/'
            self.compression_model.sample_rate = 16000
            gen_audio = self.compression_model.decode(gen_tokens, None)
            gen_audio = gen_audio.cpu()
            for i in range(len(description_list)):
                print(save_path + (file_name_list[i]))
                torchaudio.save(
                    save_path + (file_name_list[i]), gen_audio[i, :, :160000], self.sample_rate)

    description_list = []
    file_name_list = []

import os
import pandas as pd

class_ids = {
    'real': 0,
    'fake': 1
}

# -----------------------------------------------------------------------------------------
# ljspeech
# -----------------------------------------------------------------------------------------
real_entries = os.listdir('/Users/jakegus/Downloads/LJSpeech-1.1/wavs')
real_list = []
for sound in real_entries:
    real_data = {'file path': '/Users/jakegus/Downloads/LJSpeech-1.1/wavs/' + sound, 'class': class_ids['real'] }
    real_list.append(real_data)

# -----------------------------------------------------------------------------------------
# ljspeech_melgan
# -----------------------------------------------------------------------------------------
fake_entries = os.listdir('/Users/jakegus/Downloads/generated_audio/ljspeech_melgan')
train_list = []
for sound in fake_entries:
    fake_data = {'file path': '/Users/jakegus/Downloads/generated_audio/ljspeech_melgan/' + sound, 'class': class_ids['fake'] }
    train_list.append(fake_data)

# -----------------------------------------------------------------------------------------
# ljspeech_waveglow
# -----------------------------------------------------------------------------------------
fake_entries = os.listdir('/Users/jakegus/Downloads/generated_audio/ljspeech_waveglow')
fake_list = []
for sound in fake_entries:
    fake_data = {'file path': '/Users/jakegus/Downloads/generated_audio/ljspeech_waveglow/' + sound, 'class': class_ids['fake'] }
    fake_list.append(fake_data)

# -----------------------------------------------------------------------------------------
# ljspeech_full_band_melgan
# -----------------------------------------------------------------------------------------
fake_entries = os.listdir('/Users/jakegus/Downloads/generated_audio/ljspeech_full_band_melgan')
fake_list = []
for sound in fake_entries:
    fake_data = {'file path': '/Users/jakegus/Downloads/generated_audio/ljspeech_full_band_melgan/' + sound, 'class': class_ids['fake'] }
    fake_list.append(fake_data)

# -----------------------------------------------------------------------------------------
# ljspeech_hifiGAN
# -----------------------------------------------------------------------------------------
fake_entries = os.listdir('/Users/jakegus/Downloads/generated_audio/ljspeech_hifiGAN')
fake_list = []
for sound in fake_entries:
    fake_data = {'file path': '/Users/jakegus/Downloads/generated_audio/ljspeech_hifiGAN/' + sound, 'class': class_ids['fake'] }
    fake_list.append(fake_data)

# -----------------------------------------------------------------------------------------
# ljspeech_multi_band_melgan
# -----------------------------------------------------------------------------------------
fake_entries = os.listdir('/Users/jakegus/Downloads/generated_audio/ljspeech_multi_band_melgan')
fake_list = []
for sound in fake_entries:
    fake_data = {'file path': '/Users/jakegus/Downloads/generated_audio/ljspeech_multi_band_melgan/' + sound, 'class': class_ids['fake'] }
    fake_list.append(fake_data)

# -----------------------------------------------------------------------------------------
# ljspeech_parallel_wavegan
# -----------------------------------------------------------------------------------------
fake_entries = os.listdir('/Users/jakegus/Downloads/generated_audio/ljspeech_parallel_wavegan')
fake_list = []
for sound in fake_entries:
    fake_data = {'file path': '/Users/jakegus/Downloads/generated_audio/ljspeech_parallel_wavegan/' + sound, 'class': class_ids['fake'] }
    fake_list.append(fake_data)

train_real = real_list[0:6549]
test_real = real_list[6550:]

train_data = train_real + train_list
train_df = pd.DataFrame(data = train_data)

test_data = test_real + fake_list
test_df = pd.DataFrame(data = test_data)


import os
import pandas as pd

class_ids = {
    'real': 0,
    'fake': 1
}
real_list = []
train_list = []
fake_list = []
test_list = []

# -----------------------------------------------------------------------------------------
# ljspeech
# -----------------------------------------------------------------------------------------
real_entries = os.listdir('/scratch/us450/GenAudioProject/data/LJSpeech-1.1/wavs')
for sound in real_entries:
    real_data = {'file path': '/scratch/us450/GenAudioProject/data/LJSpeech-1.1/wavs/' + sound, 'class': class_ids['real'] }
    real_list.append(real_data)

# -----------------------------------------------------------------------------------------
# ljspeech_melgan
# -----------------------------------------------------------------------------------------
fake_entries = os.listdir('/scratch/us450/GenAudioProject/data/genData/generated_audio/ljspeech_melgan')
for sound in fake_entries:
    fake_data = {'file path': '/scratch/us450/GenAudioProject/data/genData/generated_audio/ljspeech_melgan/' + sound, 'class': class_ids['fake'] }
    fake_list.append(fake_data)

# -----------------------------------------------------------------------------------------
# ljspeech_waveglow
# -----------------------------------------------------------------------------------------
fake_entries = os.listdir('/scratch/us450/GenAudioProject/data/genData/generated_audio/ljspeech_waveglow')
for sound in fake_entries:
    fake_data = {'file path': '/scratch/us450/GenAudioProject/data/genData/generated_audio/ljspeech_waveglow/' + sound, 'class': class_ids['fake'] }
    fake_list.append(fake_data)

# -----------------------------------------------------------------------------------------
# ljspeech_full_band_melgan
# -----------------------------------------------------------------------------------------
fake_entries = os.listdir('/scratch/us450/GenAudioProject/data/genData/generated_audio/ljspeech_full_band_melgan')
for sound in fake_entries:
    fake_data = {'file path': '/scratch/us450/GenAudioProject/data/genData/generated_audio/ljspeech_full_band_melgan/' + sound, 'class': class_ids['fake'] }
    fake_list.append(fake_data)

# -----------------------------------------------------------------------------------------
# ljspeech_hifiGAN
# -----------------------------------------------------------------------------------------
fake_entries = os.listdir('/scratch/us450/GenAudioProject/data/genData/generated_audio/ljspeech_hifiGAN')
for sound in fake_entries:
    fake_data = {'file path': '/scratch/us450/GenAudioProject/data/genData/generated_audio/ljspeech_hifiGAN/' + sound, 'class': class_ids['fake'] }
    fake_list.append(fake_data)

# -----------------------------------------------------------------------------------------
# ljspeech_multi_band_melgan
# -----------------------------------------------------------------------------------------
fake_entries = os.listdir('/scratch/us450/GenAudioProject/data/genData/generated_audio/ljspeech_multi_band_melgan')
for sound in fake_entries:
    fake_data = {'file path': '/scratch/us450/GenAudioProject/data/genData/generated_audio/ljspeech_multi_band_melgan/' + sound, 'class': class_ids['fake'] }
    fake_list.append(fake_data)

# -----------------------------------------------------------------------------------------
# ljspeech_parallel_wavegan
# -----------------------------------------------------------------------------------------
fake_entries = os.listdir('/scratch/us450/GenAudioProject/data/genData/generated_audio/ljspeech_parallel_wavegan')
for sound in fake_entries:
    fake_data = {'file path': '/scratch/us450/GenAudioProject/data/genData/generated_audio/ljspeech_parallel_wavegan/' + sound, 'class': class_ids['fake'] }
    test_list.append(fake_data)

train_real = real_list[0:9824]
test_real = real_list[9825:]

train_data = train_real + fake_list
train_df = pd.DataFrame(data = train_data)

test_data = test_real + test_list
test_df = pd.DataFrame(data = test_data)

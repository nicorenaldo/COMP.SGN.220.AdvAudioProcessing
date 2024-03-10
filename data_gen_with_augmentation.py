
from typing import List

import random
import pathlib
import numpy as np
import soundfile as sf

from feature_extraction import extract_spectrogram
from utils import (
    get_audio_files_from_subdirs,
    get_audio_file_data,
    get_files_from_dir_with_pathlib,
    move_files,
    write_pickle,
    split_audio_into_sequences
)
from augmentation import augment_data


def main(dataset_dirs: List[pathlib.Path], output_dir: pathlib.Path, ratio: List[float]):
    n_seq = 10
    song_count = 0
    seq_count = 0

    output_combined_dir = output_dir / 'combined'
    output_combined_dir.mkdir(exist_ok=True)

    labels = [dir.name for dir in dataset_dirs]
    with open(output_dir / 'labels.txt', 'w') as f:
        f.write('\n'.join(labels))

    for dataset_dir in dataset_dirs:
        audio_paths = get_audio_files_from_subdirs(dataset_dir)
        print(f'Found {len(audio_paths)} audio files from {dataset_dir}')
        for audio_path in audio_paths:
            try:
                song_count += 1
                audio_data, sr = get_audio_file_data(audio_path)

                for seq_idx, seq in enumerate(split_audio_into_sequences(audio_data, n_seq)):
                    seq_count += 1
                    file_name = f'{audio_path.stem}_{audio_path.parent.stem}_seq_{seq_idx:03}.wav'
                    sf.write(output_combined_dir / file_name, seq, sr)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
                    

    print(f"Finished processing {song_count} songs and {seq_count} sequences.")

    print()
    print("Start shuffling and splitting the data into train, validation, and test sets.")
    all_files = list(output_combined_dir.glob('*.wav'))
    random.shuffle(all_files)

    # Calculate the number of files for each set
    num_train = int(len(all_files) * ratio[0])
    num_val = int(len(all_files) * ratio[1])

    train_files = all_files[:num_train]
    val_files = all_files[num_train:num_train + num_val]
    test_files = all_files[num_train + num_val:]

    # Move the files
    move_files(train_files, output_dir / 'train')
    move_files(val_files, output_dir / 'val')
    move_files(test_files, output_dir / 'test')

    print(f"Finished splitting data")
    print(f"Train: {len(train_files)} files")
    print(f"Validation: {len(val_files)} files")
    print(f"Test: {len(test_files)} files")

    output_combined_dir.rmdir()

    ## Serialize
    print("\nStart augmenting training data and serializing all data splits.")
    split = ['train', 'val', 'test']
    
    for s in split:
        print(f"Processing {s} data...")
        all_files = get_files_from_dir_with_pathlib(output_dir / s)
        
        for audio_file in all_files:
            try:
                audio_data, sr = get_audio_file_data(audio_file)
                stft = extract_spectrogram(audio_data)

                one_hot_label = np.zeros(len(labels), dtype=np.float32)
                classname = audio_file.name.partition('.')[0]
                one_hot_label[labels.index(classname)] = 1
                
                file_name = f'{audio_file.stem}'

                features_and_metadata = {
                    'features': stft,
                    'label': one_hot_label
                }
                write_pickle(
                    str(audio_file.parent / f'{file_name}.pkl'),
                    features_and_metadata)
                
                if s == 'train':
                    augment_data(audio_data, file_name, one_hot_label, audio_file.parent)

            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue

            audio_file.unlink()
        
        print(f"Finished processing {s} data\n")


if __name__ == '__main__':
    dataset_root_path = pathlib.Path("dataset/raw_audio")
    dataset_dirs = get_files_from_dir_with_pathlib(dataset_root_path)

    output_dir = pathlib.Path('dataset/features')
    if output_dir.exists():
        for dir in output_dir.iterdir():
            if dir.is_dir():
                for file in dir.iterdir():
                    file.unlink()
                dir.rmdir()

    output_dir.mkdir(exist_ok=True)

    ratio = [0.8, 0.1, 0.1]  # train, test, validation
    main(dataset_dirs, output_dir, ratio)

import os
from tqdm import tqdm

def process_file(filename):
    segments = []
    cls = None  
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) < 5:
                continue  
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            x_min = x - (w / 2)
            y_min = y - (h / 2)
            x_max = x + (w / 2)
            y_max = y + (h / 2)
            segment = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            segments.append(segment)
    return cls, segments

def save_segments(filename, cls, segments):
    with open(filename, 'w') as file:
        for segment in segments:
            line = f'{cls} ' + ' '.join([f'{coord:.6f}' for coord in segment])
            file.write(line + '\n')

def main():
    input_dir = '/home/pranav/Projects/arrow_detection/data/labels/test'
    output_dir = '/home/pranav/Projects/arrow_detection/data/masks/test'
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_dir, filename)
            try:
                cls, segments = process_file(filepath)
            except Exception:
                segments = []
            new_filepath = os.path.join(output_dir, filename)
            if not segments:
                open(new_filepath, 'w').close()
            else:
                save_segments(new_filepath, cls, segments)

if __name__ == '__main__':
    main()

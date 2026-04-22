import csv
import json
import os

input_dir = r'./opus4.7_JSON'
output_csv = r'./opus4.7_classified.csv'

json_file = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])

fieldnames = [
    'video_id',
    'is_poisoned',
    'attack_level',
    'semantic',
    'logical',
    'decision',
    'final_score',
    'reasoning'
]

with open(output_csv, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for file_name in json_file:
        file_path = os.path.join(input_dir, file_name)

        with open(file_path, 'r', encoding='utf-8') as fin:
            data = json.load(fin)

        row = {
            'video_id': data['video_id'],
            'is_poisoned': str(data['is_poisoned']).upper(),
            'attack_level': data['attack_level'],
            'semantic': data['scores']['semantic'],
            'logical': data['scores']['logical'],
            'decision': data['scores']['decision'],
            'final_score': data['final_score'],
            'reasoning': data['reasoning']
        }

        writer.writerow(row)

print('Done. CSV saved to:', output_csv)
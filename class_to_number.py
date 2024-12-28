classes = [
    'A-1',
    'A-2',
    'A-3',
    'A-4',
    'A-6a',
    'A-6b',
    'A-6c',
    'A-6d',
    'A-6e',
    'A-7',
    'A-8',
    'A-16',
    'A-17',
    'A-18b',
    'A-21',
    'A-29',
    'A-30',
    'A-32',
    'B-1',
    'B-2',
    'B-20',
    'B-21',
    'B-22',
    'B-23',
    'B-33',
    'B-36',
    'B-41',
    'B-43',
    'C-2',
    'C-4',
    'C-5',
    'C-9',
    'C-10',
    'C-12',
    'D-1',
    'D-2',
    'D-3',
    'D-4a',
    'D-6',
    'D-6b',
    'D-7',
    'D-8',
    'D-14',
    'D-15',
    'D-18',
    'D-23',
    'D-28',
    'D-29',
    'D-42',
    'D-43'
]

class_to_number = {cls: idx for idx, cls in enumerate(classes)}

number_to_class = {idx: cls for cls, idx in class_to_number.items()}

if __name__ == "__main__":
    print("CLASSES NAMES TO NUMBERS MAP")
    print("Classes:")
    for c in classes:
        print(f"\t- '{c}' with number {class_to_number[c]}")

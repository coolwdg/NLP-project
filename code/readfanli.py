import random


def read_qa_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    texts = content.split('[TEXT]')[1:]
    all_qa_pairs = []
    for text in texts:
        text_lines = text.strip().split('\n')
        text_content = text_lines[0].strip() if text_lines else ""
        current_qa = []
        for line in text_lines:
            if line.startswith('[QUESTION]'):
                question = line.replace('[QUESTION]', '').strip()
            elif line.startswith('[ANSWER]'):
                answer = line.replace('[ANSWER]', '').strip()
                current_qa.append((question, answer, text_content))
        all_qa_pairs.extend(current_qa)
    return all_qa_pairs


def select_and_save(qa_pairs, output_file):
    selected_qa = random.sample(qa_pairs, 30)
    with open(output_file, 'w', encoding='utf-8') as file:
        for question, _, text in selected_qa:
            file.write(f"[TEXT] {text}\n")
            file.write(f"[QUESTION] {question}\n\n")


if __name__ == "__main__":
    input_file = 'qa_data.txt'
    output_file = 'fanli.txt'
    qa_pairs = read_qa_data(input_file)
    select_and_save(qa_pairs, output_file)
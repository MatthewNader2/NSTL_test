input_source = 'sample_input.txt'
with open(input_source, 'r', encoding='utf-8') as _fh:
    raw_text = _fh.read()
cleaned_text = raw_text.strip()
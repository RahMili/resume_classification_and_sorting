import docx


def read_docx_file(data_path):
    # Open the docx file
    doc = docx.opendocx(data_path)

    # Iterate through all paragraphs in the document
    text = ''
    for para in doc.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        text = text + str(''.join(node.text for node in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')))

    return text

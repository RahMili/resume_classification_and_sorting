import PyPDF2


def read_native_pdf(pdf_file_path):
    pdf_file = open(pdf_file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    text = ''

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        # print('\n\n\nTEXT:', page_text)
        text = text + page_text

    #print(text)
    return text

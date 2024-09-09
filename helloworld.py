from unstructured.partition.pdf import partition_pdf
import pypdf


doc_list = partition_pdf(filename= "input_pdf.pdf")
doc_list[0]

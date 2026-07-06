"""提取PDF文本内容"""
import pypdfium2 as pdfium

pdf_path = r"d:\tools\workspace_pycharm\SDDiP-RL\modeling\doc\xunhangsun-1-s2.0-S2352152X24007096-main(1).pdf"
output_path = r"d:\tools\workspace_pycharm\SDDiP-RL\modeling\doc\pdf_extracted3.txt"

pdf = pdfium.PdfDocument(pdf_path)
print(f"Total pages: {len(pdf)}")

all_text = []
for i, page in enumerate(pdf):
    textpage = page.get_textpage()
    text = textpage.get_text_range()
    all_text.append(f"===== PAGE {i+1} =====\n{text}")
    print(f"Page {i+1} done, {len(text)} chars")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_text))

print(f"Saved to {output_path}")

import fitz

doc = fitz.open("MOHS_MICROGRAPHIC_SURGERY_from_layers_to_reconstruction.pdf")   
print("Number of pages:", doc.page_count)

pages = []  # page[0] is book's page 1 text...

for page in doc:
    pages.append(page.get_text())
    
print("Pages collected:", len(pages))
print("Characters in page 101:", len(pages[100]))

print(pages[100])

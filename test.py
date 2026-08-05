import fitz
import json


PDF = "MOHS_MICROGRAPHIC_SURGERY_from_layers_to_reconstruction.pdf"

def clean(text):
    return " ".join(text.split())

doc = fitz.open(PDF)

chunks = []
for page_number, page in enumerate(doc):
    for b in page.get_text("blocks"): #b is tuple(x0, y0, x1, y1, text, block_no, block_type)
        text = clean(b[4])
        if len(text) > 100:
            chunks.append({"text": text, "page": page_number + 1})

print("Total chunks:", len(chunks))
print("Chunks per page:", round(len(chunks) / doc.page_count, 1))


#save chunks in json
with open("chunks.json", "w") as f:
    json.dump(chunks, f)
print("Saved", len(chunks), "chunks")
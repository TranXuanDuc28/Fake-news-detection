import pypdf
import sys

def extract_text(pdf_path, output_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        print(f"Total pages: {len(reader.pages)}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Total pages: {len(reader.pages)}\n")
            for idx, page in enumerate(reader.pages):
                f.write(f"\n--- Page {idx + 1} ---\n")
                f.write(page.extract_text() or "")
        print("Done writing to text file.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_text("d:\\XuanDuc\\TaiLieuKi8\\CuoiKiHocSau\\miniproject.pdf", "miniproject_text.txt")

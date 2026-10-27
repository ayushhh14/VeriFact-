from src.ingestion import chunk_pages


def test_chunk_pages_preserves_page_and_source():
    pages = [{"source": "paper.pdf", "page": 3, "text": " ".join(f"word{i}" for i in range(25))}]
    chunks = chunk_pages(pages, chunk_size=10, overlap=2)
    assert len(chunks) == 3
    assert all(chunk["source"] == "paper.pdf" for chunk in chunks)
    assert all(chunk["page"] == 3 for chunk in chunks)
    assert "word0" in chunks[0]["text"]
    assert "word24" in chunks[-1]["text"]

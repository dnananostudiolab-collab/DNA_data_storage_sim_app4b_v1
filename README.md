# Raw Representation DNA Storage App

This Streamlit app stores image, text, and audio as raw/canonical data before DNA encoding.

Pipeline:

1. Input
2. Compression (raw representation stage)
3. Encoding
4. Strand Design
5. Decoding
6. Summarization

Two tabs are included:

- No ECC baseline
- Reed-Solomon ECC

Supported raw representations:

- Image: RGB pixels, grayscale pixels, binary pixels
- Text: UTF-8 bytes
- Audio: PCM16 waveform rebuilt as WAV

Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Notes:

- Reed-Solomon protects the raw bytes before SM/R∞ DNA encoding.
- Substitution-only errors are recommended first. Insertion/deletion changes DNA length and can break SM/R∞ framing.
- Panel 5 is for decode and output preview. Panel 6 is a compact summarization panel with only the essential evidence for successful decoding.

# data/

Place your five 16-bit grayscale depth PNGs here:

```
D1_16.png
D2_16.png
D3_16.png
D4_16.png
D5_16.png
```

Each must be a single-channel 16-bit PNG (pixel values 0..65535), loaded with
`cv2.IMREAD_UNCHANGED`. The images are **not** included in this repository;
supply your own. You can point the experiment scripts at a different directory
with `--data-dir`, or override the file names via `src/io_utils.py`.

The `.gitignore` excludes `data/` (except this note) so your images are never
committed.
